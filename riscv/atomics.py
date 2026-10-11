"""Atomic LR/SC and AMOs across harts that run in parallel host threads.

In vcml-pydrofoil every hart is its own Pydrofoil instance (own dlmopen
namespace, own thread with async=true) and all of them access the same guest
RAM through DMI, i.e. through the same host memory. Sail executes the A
extension in several steps (read - compute - write for an AMO, check
reservation - write for an SC) and keeps the LR/SC reservation private per
model, so neither is atomic across harts.

The fix does the critical step with an atomic host instruction on the guest
RAM, the same approach as QEMU's multi-threaded TCG. patch_execute wraps the
execute functions of LOADRES (LR), STORECON (SC) and AMO: while one of them
runs, g.atomic_kind is set, and the memory write hooks hand the (only) store
of the instruction to atomic_write:

  LR   unchanged; afterwards the wrapper remembers the loaded value (rd) as
       reservation_value
  SC   Sail checks the reservation (same address as the LR) as before; the
       store becomes compare-and-swap(addr, reservation_value, new): it only
       happens if memory still holds what LR loaded. Check and store are one
       host instruction, so two harts can never both succeed, and a store of
       another hart in between makes it fail. On failure the wrapper sets
       rd = 1 (Sail itself always writes 0 after a store that did not trap)
  AMO  Sail reads and computes as before, but its store is replaced by the
       complete read-modify-write as one host atomic (fetch_add, exchange, ...
       / CAS loop for min/max) on the operand rs2. The wrapper then writes the
       old value returned by the host atomic to rd, replacing the one Sail read

Only the store is touched, never a read, because a read during the
instruction may also be a page table walk. The store is the data store as
long as hardware A/D bit updates are off (rv_enable_dirty_update, default).

Whether the store goes through write_mem or write_mem_exclusive depends on the
.aq/.rl bits, so both hooks check the flag.

Without a host address (MMIO, or a memory without host_address) the old,
non-atomic Sail behaviour stays. The host address comes from
MemBase.host_address() (pydrofoil/mem.py), which returns 0 by default;
MemoryObserver (interp_plugin.py) answers it for the regions registered with
_add_host_memory.

Known gap: a value changed and changed back between LR and SC (A -> B -> A)
is not noticed, which the specification forbids. Irrelevant for locks,
counters and C11/Linux cmpxchg (QEMU accepts the same).
"""

from rpython.rlib.rarithmetic import r_uint, intmask
from rpython.rlib.unroll import unrolling_iterable
from rpython.rlib import jit
from rpython.rtyper.lltypesystem import rffi, lltype
from rpython.translator.tool.cbuild import ExternalCompilationInfo


# ---------------------------------------------------------------------------
# host atomics, in C because RPython has no atomic RMW operations

_C_SOURCE = r"""
#include <stdint.h>

/* RISC-V AMO funct5 values */
enum { AMOADD = 0x00, AMOSWAP = 0x01, AMOXOR = 0x04, AMOOR = 0x08,
       AMOAND = 0x0c, AMOMIN = 0x10, AMOMAX = 0x14, AMOMINU = 0x18,
       AMOMAXU = 0x1c };

/* store desired at p if p still holds expected; returns 1 on success */
RPY_EXTERN long pyd_cas(void* p, long n, uint64_t expected, uint64_t desired)
{
    if (n == 4) {
        uint32_t e = (uint32_t)expected;
        return __atomic_compare_exchange_n((uint32_t*)p, &e, (uint32_t)desired,
                                           0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
    }
    uint64_t e = expected;
    return __atomic_compare_exchange_n((uint64_t*)p, &e, desired,
                                       0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
}

#define PYD_AMO(T, ST)                                                        \
    static uint64_t pyd_amo_##T(T* p, long funct5, T op)                      \
    {                                                                         \
        T old, new_;                                                          \
        switch (funct5) {                                                     \
        case AMOSWAP: return __atomic_exchange_n(p, op, __ATOMIC_SEQ_CST);    \
        case AMOADD:  return __atomic_fetch_add(p, op, __ATOMIC_SEQ_CST);     \
        case AMOXOR:  return __atomic_fetch_xor(p, op, __ATOMIC_SEQ_CST);     \
        case AMOOR:   return __atomic_fetch_or(p, op, __ATOMIC_SEQ_CST);      \
        case AMOAND:  return __atomic_fetch_and(p, op, __ATOMIC_SEQ_CST);     \
        }                                                                     \
        /* min/max: no host instruction, CAS loop */                          \
        old = __atomic_load_n(p, __ATOMIC_SEQ_CST);                           \
        do {                                                                  \
            switch (funct5) {                                                 \
            case AMOMIN:  new_ = (ST)old < (ST)op ? old : op; break;          \
            case AMOMAX:  new_ = (ST)old > (ST)op ? old : op; break;          \
            case AMOMINU: new_ = old < op ? old : op; break;                  \
            default:      new_ = old > op ? old : op; break; /* AMOMAXU */    \
            }                                                                 \
        } while (!__atomic_compare_exchange_n(p, &old, new_, 0,               \
                                              __ATOMIC_SEQ_CST,               \
                                              __ATOMIC_SEQ_CST));             \
        return old;                                                           \
    }

PYD_AMO(uint32_t, int32_t)
PYD_AMO(uint64_t, int64_t)

/* the complete AMO, returns the old memory value (zero-extended) */
RPY_EXTERN uint64_t pyd_amo(void* p, long n, long funct5, uint64_t operand)
{
    if (n == 4)
        return pyd_amo_uint32_t((uint32_t*)p, funct5, (uint32_t)operand);
    return pyd_amo_uint64_t((uint64_t*)p, funct5, operand);
}

"""

_eci = ExternalCompilationInfo(
    separate_module_sources=[_C_SOURCE],
    post_include_bits=[
        "#include <stdint.h>\n"
        "RPY_EXTERN long pyd_cas(void* p, long n, uint64_t expected, uint64_t desired);\n"
        "RPY_EXTERN uint64_t pyd_amo(void* p, long n, long funct5, uint64_t operand);\n"
    ],
)

def _external(name, args, result):
    return rffi.llexternal(name, args, result, compilation_info=_eci,
                           _nowrapper=True, sandboxsafe=True)

_c_cas = _external("pyd_cas", [rffi.VOIDP, rffi.LONG, rffi.ULONGLONG, rffi.ULONGLONG], rffi.LONG)
_c_amo = _external("pyd_amo", [rffi.VOIDP, rffi.LONG, rffi.LONG, rffi.ULONGLONG], rffi.ULONGLONG)

# ---------------------------------------------------------------------------
# decoding the current instruction

KIND_NONE = 0   # no A-extension instruction running
KIND_LR = 1
KIND_SC = 2
KIND_AMO = 3

def funct5(instbits):
    return intmask((r_uint(instbits) >> 27) & 0x1f)

def rs2(instbits):
    return intmask((r_uint(instbits) >> 20) & 0x1f)

def rd(instbits):
    return intmask((r_uint(instbits) >> 7) & 0x1f)

# x1..x31, unrolled by the translator in Machine.atomic_xreg
# (supportcoderiscv.py), which reads the register attributes by name
XREGS = unrolling_iterable(range(1, 32))

def amo_result(old, instbits, rv64):
    """rd value of an AMO: the old memory value, sign-extended for .W"""
    if not rv64:
        return old & r_uint(0xffffffff)
    if (r_uint(instbits) >> 12) & 0x7 == 2 and old & r_uint(0x80000000):  # .W
        return old | (r_uint(0xffffffff) << 32)
    return old

# ---------------------------------------------------------------------------
# the store, called from write_mem / write_mem_exclusive while g.atomic_kind
# is set

@jit.dont_look_inside
def atomic_write(g, addr, n, value):
    """Performs the store of the running SC / AMO atomically. Returns False
    if the normal store has to happen instead (no host memory, i.e. MMIO)."""
    ptr = g.mem.host_address(addr, n)
    if not ptr:
        return False
    p = rffi.cast(rffi.VOIDP, ptr)
    if g.atomic_kind == KIND_SC:
        g.atomic_failed = _c_cas(p, n, g.reservation_value, value) == 0
        return True
    if g.atomic_kind == KIND_AMO:
        g.amo_old = r_uint(_c_amo(p, n, g.amo_op, g.amo_operand))
        g.amo_done = True
        return True
    return False

# ---------------------------------------------------------------------------
# the wrappers around the execute functions

def patch_execute(outriscv, rv64):
    for cls, kind in [(outriscv.Union_zast_zLOADRES, KIND_LR),
                      (outriscv.Union_zast_zSTORECON, KIND_SC),
                      (outriscv.Union_zast_zAMO, KIND_AMO)]:
        _patch_execute(outriscv, rv64, cls, kind)

def _patch_execute(outriscv, rv64, cls, kind):
    orig = cls.meth_zexecute
    def execute(self, machine):
        g = machine.g
        instbits = machine.atomic_instbits()
        g.atomic_kind = kind
        g.atomic_failed = False
        g.amo_done = False
        if kind == KIND_AMO:
            g.amo_op = funct5(instbits)
            g.amo_operand = machine.atomic_xreg(rs2(instbits))
        res = orig(self, machine)
        g.atomic_kind = KIND_NONE
        if machine.have_exception:
            return res
        r = rd(instbits)
        if kind == KIND_LR:
            g.reservation_value = machine.atomic_xreg(r)
        elif kind == KIND_SC:
            if g.atomic_failed:
                outriscv.func_zwX(machine, r, r_uint(1))
        else:
            if g.amo_done:
                outriscv.func_zwX(machine, r, amo_result(g.amo_old, instbits, rv64))
        return res
    execute.func_name = "atomic_execute_" + cls.__name__
    cls.meth_zexecute = execute
