from dataclasses import dataclass, field
from parser import (
    BinaryOpExpr,
    DataExpr,
    Expr,
    LiteralExpr,
    NameExpr,
    NAryMessageExpr,
    ParenExpr,
    QuoteExpr,
    SequenceExpr,
    TupleExpr,
    UnaryMessageExpr,
    UnaryOpExpr,
)
from typing import Callable, Optional, Tuple, TypeAlias, TypeVar, Union

from termcolor import colored

from interpreter import (
    BytecodeOp,
    BytecodeSequence,
    CompiledBody,
    CompileTimeHandler,
    Context,
    IntrinsicHandler,
    IntrinsicMethodBody,
    JumpIndex,
    Method,
    MultiMethod,
    NativeHandler,
    NativeMethodBody,
    NullValue,
    NumberValue,
    ParameterAnyMatcher,
    ParameterMatcher,
    ParameterTypeMatcher,
    ParameterValueMatcher,
    QuoteMethodBody,
    QuoteType,
    QuoteValue,
    StringValue,
    SymbolValue,
    TupleType,
    TypeValue,
    Value,
    VectorType,
    is_subtype,
    type_of,
)
from lexer import TokenType
from span import SourceSpan


# Singleton.
@dataclass
class DefaultReceiver:
    pass


default_receiver = DefaultReceiver()


@dataclass
class CompilationContext:
    # Indicates where each slot value may be found.
    slots: list[Tuple[Union[str, DefaultReceiver], "Register"]]
    base: Union[Context, "CompilationContext"]

    def __repr__(self):
        return "<CompilationContext>"

    @staticmethod
    def from_context(ctxt: Context) -> "CompilationContext":
        assert isinstance(ctxt, Context)
        return CompilationContext(slots=[], base=ctxt)

    @staticmethod
    def stacked_compilation_context(
        slots: list, base: Union[Context, "CompilationContext"]
    ) -> "CompilationContext":
        assert isinstance(base, (Context, CompilationContext))
        ## Just add all the slots already on the "stack". This wastes some memory, but
        ## makes it easy to determine slot indices.
        # return CompilationContext(slots=list(ctxt.slots), base=ctxt)
        return CompilationContext(slots=list(slots), base=base)

    # If the name is found within compilation-context slots, returns the register holding the slot value.
    # Otherwise, if the name is found within regular-context slots, returns that value.
    # Otherwise, returns None.
    # Example:
    #   For the following compilation context:
    #       <top> CompilationContext
    #           default_receiver: [2]
    #           "local-var":      [3]
    #       <base> CompilationContext
    #           default_receiver: [0]
    #           "outer-var":      [1]
    #       <base> Context
    #           "global-var": <some value>
    #           ...
    #   we have the following lookup results:
    #       default_receiver -> [2]
    #       "local-var" -> [3]
    #       "outer-var" -> [1]
    #       "global-var" -> <some value>
    #       "nonexistent-var" -> None
    def lookup(
        self, name: Union[str, DefaultReceiver]
    ) -> Optional[Union["Register", Value, MultiMethod, CompileTimeHandler]]:
        ctxt = self
        while isinstance(ctxt, CompilationContext):
            for slot, value in ctxt.slots:
                if name == slot:
                    return value
            ctxt = ctxt.base
        # Just look up in the regular Context.
        assert isinstance(ctxt, Context)
        assert (
            name is not default_receiver
        ), "shouldn't have gotten here... someone forgot to add a default_receiver slot!"
        while ctxt:
            if name in ctxt.slots:
                return ctxt.slots[name]
            ctxt = ctxt.base
        return None


@dataclass(frozen=True)
class Register:
    index: int


@dataclass(frozen=True)
class SlotRegister(Register):
    def __str__(self):
        return f"[{self.index}]"


@dataclass(frozen=True)
class VirtualRegister(Register):
    def __str__(self):
        # return f"@{self.index}({id(self) % 1000})"
        return f"@{self.index}"

    # def repoint_to(self, other: "VirtualRegister") -> None:
    #    assert isinstance(other, VirtualRegister)
    #    self.index = other.index


# Intermediate Representation op
@dataclass(kw_only=True)
class IROp:
    span: SourceSpan
    # Some instructions don't output a result.
    dst: Optional[Register] = None
    # Example: method blocks for a multimethod dispatch, or an inlined if/then/else.
    # The `args` may hold references to these blocks; this is just a standard form
    # for other analyses to be able to use.
    sub_blocks: list["TreeIRBlock"] = field(default_factory=list)


# Not really an operation on its own, just hosts a sub-block.
@dataclass(kw_only=True)
class InlineBlockOp(IROp):
    span = None
    dst = None

    def __post_init__(self):
        assert not self.dst
        assert len(self.sub_blocks) == 1


# Not really an operation, just a jump target.
@dataclass(kw_only=True)
class Label(IROp):
    id: int

    def __post_init__(self):
        assert not self.dst

    def __hash__(self):
        return hash(self.id)


@dataclass(kw_only=True)
class JumpOp(IROp):
    target: Label

    def __post_init__(self):
        assert not self.dst


@dataclass(kw_only=True)
class UnreachableOp(IROp):
    # May have a destination register, as a stand-in.
    # Dead code elimination should end up deleting all instances of such a register.
    pass


@dataclass(kw_only=True)
class ReturnOp(IROp):
    value: Register

    def __post_init__(self):
        assert not self.dst


@dataclass(kw_only=True)
class CopyOp(IROp):
    src: Register


# CopyOp, except the value copied to destination is selected based on which block
# was just executed.
@dataclass(kw_only=True)
class PhiOp(IROp):
    # Each source register is associated with the instruction which computes it, or the jump
    # op indicating which predecessor block lead to this phi op, or None for a slot register
    # (which simply exists on method entry). (None may also be used to indicate that there is
    # no ambiguity in what the source is, and the source instruction should simply be looked
    # up by source register.)
    srcs: list[Tuple[Register, Optional[IROp]]]
    # If true, this phi op must not be optimized away, even if there is only a single source.
    # This is effectively an "open" phi, where additional sources may be added later (due to
    # tail-recursion).
    force_keep: bool = False


@dataclass(kw_only=True)
class LiteralOp(IROp):
    value: Value


@dataclass(kw_only=True)
class BaseInvokeOp(IROp):
    call_args: list[Register]
    tail_call: bool
    tail_position: bool

    def __post_init__(self):
        # Require a destination register, even if tail-call. This is for a couple reasons:
        # - Multimethod dispatch ops must have a destination in case of dispatch failure.
        # - Other invocations may get inlined, which means that they are no longer tail calls at all.
        assert self.dst is not None


@dataclass(kw_only=True)
class InvokeRegisterOp(BaseInvokeOp):
    callable: Register


@dataclass(kw_only=True)
class InvokeMultimethodOp(BaseInvokeOp):
    multimethod: MultiMethod


@dataclass(kw_only=True)
class InvokeMethodOp(BaseInvokeOp):
    method: Method
    method_start_label: Label


@dataclass(kw_only=True)
class InvokeQuoteOp(BaseInvokeOp):
    quote: QuoteMethodBody


@dataclass(kw_only=True)
class InvokeIntrinsicOp(BaseInvokeOp):
    intrinsic: IntrinsicHandler


@dataclass(kw_only=True)
class InvokeNativeOp(BaseInvokeOp):
    native: NativeHandler
    return_type: Optional[TypeValue]


@dataclass(kw_only=True)
class ClosureOp(IROp):
    quote: QuoteValue


@dataclass(kw_only=True)
class SlotLookupOp(IROp):
    slot_name: str


@dataclass(kw_only=True)
class VectorOp(IROp):
    components: list[Register]


@dataclass(kw_only=True)
class TupleOp(IROp):
    components: list[Register]


@dataclass(kw_only=True)
class MultimethodDispatchOp(IROp):
    # For debug
    slot_name: str
    dispatch_args: list[Register]
    dispatch: list[Tuple[Method, "TreeIRBlock"]]
    no_matching_method: Optional["TreeIRBlock"]
    ambiguous_method_resolution: Optional["TreeIRBlock"]


@dataclass(kw_only=True)
class SignalOp(IROp):
    condition_name: str
    signal_args: list[Register]


@dataclass(kw_only=True)
class IfElseOp(IROp):
    condition: Register
    true_block: "TreeIRBlock"
    false_block: "TreeIRBlock"


@dataclass
class InliningInfo:
    body: QuoteMethodBody
    allows_tail_recursion: bool
    # Everything below here is required if allows_tail_recursion, and must be None otherwise.
    # ---------------------------------------------------------------------------------------
    # Phi op per <default-receiver> and method parameter/argument.
    # The body must have been compiled to use these outputs as the argument values.
    argument_phi_ops: Optional[list[PhiOp]]
    entry_point: Optional[Label]


@dataclass
class TreeIRBlock:
    ops: list[IROp]
    # Context to use when compiling / optimizing anything in the body of this block.
    ctxt: CompilationContext
    # Stack of (information about) quote-method-bodies we have inlined so far in the static scope of this block.
    inlining_stack: list[InliningInfo]

    def __repr__(self):
        return "<TreeIRBlock>"


@dataclass(kw_only=True)
class BasicBlockJumpOp(IROp):
    target: "BasicIRBlock"

    def __post_init__(self):
        assert not self.dst


# Used only temporarily, during conversion from tree IR to basic blocks.
@dataclass(kw_only=True)
class TransitionalPhiOp(IROp):
    srcs: list[Tuple[Register, Optional[IROp]]]


@dataclass(kw_only=True)
class BasicBlockPhiOp(IROp):
    srcs: list[Tuple[Register, "BasicIRBlock"]]


@dataclass(kw_only=True)
class BasicBlockMultimethodDispatchOp(IROp):
    # For debug
    slot_name: str
    # Dispatch arg may be None if its value cannot affect the dispatch result.
    dispatch_args: list[Optional[Register]]
    dispatch: list[Tuple[Method, "BasicIRBlock"]]
    no_matching_method: Optional["BasicIRBlock"]
    ambiguous_method_resolution: Optional["BasicIRBlock"]


@dataclass(kw_only=True)
class BasicBlockIfElseOp(IROp):
    condition: Register
    true_block: "BasicIRBlock"
    false_block: "BasicIRBlock"


# Singleton.
@dataclass
class AnyType:
    pass


ANY_TYPE = AnyType()

Types: TypeAlias = Union[list[TypeValue], AnyType]


# Basic block à la Single Static Assignment form.
@dataclass
class BasicIRBlock:
    # For debug only.
    id: int
    # Only the last operation may have any nonlinear control flow, i.e. any of:
    # - JumpOp (rather, BasicBlockJumpOp)
    # - ReturnOp
    # - MultimethodDispatchOp (rather, BasicBlockMultimethodDispatchOp)
    # - IfElseOp (rather, BasicBlockIfElseOp)
    # Also, there should be no label ops in here, and all inline block ops should
    # be flattened.
    ops: list[IROp]
    incoming: set["BasicIRBlock"]
    outgoing: set["BasicIRBlock"]

    # Calculated after converting to basic blocks.
    dominators: set["BasicIRBlock"]

    # Calculated essentially on-demand.
    # Per register, indicates knowledge of what possible types the value is known to have,
    # or ANY_TYPE if it could take any type. If a register is not present, we don't yet know
    # what types it may have, and must not use it for typing judgements yet.
    # Note that this implies a _subtyping_ relationship: the value is a subtype of any of these
    # provided types.
    types: Optional[dict[Register, Types]]

    def __repr__(self):
        return f"BasicIRBlock(id={self.id})"

    def __hash__(self):
        return hash(self.id)

    def link_outgoing(self, succ: "BasicIRBlock"):
        succ.incoming.add(self)
        self.outgoing.add(succ)

    def unlink_outgoing(self, succ: "BasicIRBlock"):
        succ.incoming.remove(self)
        self.outgoing.remove(succ)

    linear_op_types = (
        UnreachableOp,
        CopyOp,
        BasicBlockPhiOp,
        LiteralOp,
        BaseInvokeOp,
        ClosureOp,
        SlotLookupOp,
        VectorOp,
        TupleOp,
        SignalOp,
    )
    nonlinear_op_types = (
        BasicBlockJumpOp,
        ReturnOp,
        BasicBlockMultimethodDispatchOp,
        BasicBlockIfElseOp,
    )

    # Indicates whether the block falls through (does not have a jump or other
    # nonlinear control flow op at the end) or not.
    def falls_through(self) -> bool:
        return not (self.ops and isinstance(self.ops[-1], self.nonlinear_op_types))

    def add_op(self, op: IROp):
        # Should not be adding any more ops once there's already a nonlinear
        # control flow op.
        assert self.falls_through()
        assert isinstance(op, self.linear_op_types + self.nonlinear_op_types + (TransitionalPhiOp,))
        self.ops.append(op)

    def validate(self, is_reachable: bool):
        assert self.ops
        for op in self.ops[:-1]:
            assert isinstance(op, self.linear_op_types)
        assert isinstance(self.ops[-1], self.linear_op_types + self.nonlinear_op_types)
        for pred in self.incoming:
            assert self in pred.outgoing
        for succ in self.outgoing:
            assert self in succ.incoming

        def basic_block_has_output(block: BasicIRBlock, reg: Register):
            return any(op.dst == reg for op in block.ops)

        # All phi nodes should only have sources which are slot registers, or else the output of an
        # op in (a dominator of) a predecessor block. (Also, the source block associated to the source
        # register should itself be a predecessor, in all cases.)
        # Similarly, all non-phi nodes should only have sources which are slot registers, or else the
        # output of an op in a dominator of the _current_ block.
        if not is_reachable:
            # If the block is unreachable, skip these checks -- we're going to delete the block anyway.
            return
        for op in self.ops:
            if isinstance(op, BasicBlockPhiOp):
                for reg, block in op.srcs:
                    assert block
                    assert (
                        block in self.incoming
                    ), f"in block {self.id}: invalid phi node for {op.dst}: source {reg} is from block {block.id}, which is not a predecessor of current block {self.id}"

                    if isinstance(reg, SlotRegister):
                        continue

                    assert any(
                        basic_block_has_output(dom, reg) for dom in block.dominators
                    ), f"in block {self.id}: invalid phi node for {op.dst}: source {reg} is ostensibly from block {block.id}, but is not generated by any dominators of {block.id}"
            else:
                srcs = basic_ir_op_reg_srcs(op)
                for reg in srcs:
                    if isinstance(reg, SlotRegister):
                        continue
                    assert any(
                        basic_block_has_output(dom, reg) for dom in self.dominators
                    ), f"in block {self.id}: invalid op {op}: source {reg} is not generated by any dominators of current block {self.id}"


# Low-level Representation op.
@dataclass
class LROp:
    span: SourceSpan


@dataclass
class CopyLROp(LROp):
    dst: Register
    src: Register


@dataclass
class LiteralLROp(LROp):
    dst: Register
    value: Value


@dataclass
class PushLROp(LROp):
    src: Register


@dataclass
class PopLROp(LROp):
    dst: Register


@dataclass
class DropLROp(LROp):
    pass


@dataclass
class InvokeRegisterLROp(LROp):
    callable: Register
    num_args: int
    tail_call: bool


@dataclass
class InvokeMultimethodLROp(LROp):
    multimethod: MultiMethod
    num_args: int
    tail_call: bool


@dataclass
class InvokeIntrinsicLROp(LROp):
    intrinsic: IntrinsicHandler
    num_args: int
    tail_call: bool


@dataclass
class InvokeNativeLROp(LROp):
    native: NativeHandler
    num_args: int
    tail_call: bool


@dataclass
class ClosureLROp(LROp):
    dst: Register
    quote: QuoteValue


@dataclass
class SlotLookupLROp(LROp):
    dst: Optional[Register]
    slot_name: str


@dataclass
class VectorLROp(LROp):
    dst: Register
    components: list[Register]


@dataclass
class TupleLROp(LROp):
    dst: Register
    components: list[Register]


@dataclass
class SignalLROp(LROp):
    dst: Optional[Register]
    condition_name: str
    signal_args: list[Register]


@dataclass
class JumpLROp(LROp):
    target: Union[int, "LRBlock"]


@dataclass
class ReturnLROp(LROp):
    src: Register


@dataclass
class ConditionalJumpLROp(LROp):
    condition: Register
    match: bool
    # Jump to the target if the condition value equals the match value.
    target: Union[int, "LRBlock"]


@dataclass
class MultimethodDispatchLROp(LROp):
    # For debug
    slot_name: str
    dispatch_args: list[Register]
    # List of dispatch options, with parameter matchers (length equals that of dispatch_args)
    # and jump target if this is the best selected option.
    dispatch: list[Tuple[list[ParameterMatcher], Union[int, "LRBlock"]]]
    no_matching_method: Optional[Union[int, "LRBlock"]]
    ambiguous_method_resolution: Optional[Union[int, "LRBlock"]]


# Block of low-level representation ops. Last stop before linearizing blocks into a single
# op array.
@dataclass
class LRBlock:
    id: int
    ops: list[LROp]
    incoming: set["LRBlock"]
    outgoing: set["LRBlock"]

    def __hash__(self) -> int:
        return hash(self.id)


def should_inline_multimethod_dispatch(multimethod: MultiMethod) -> bool:
    # TODO: Inline also if heuristics suggest it should be inlined.
    return multimethod.inline_dispatch


def should_inline_quote_method(method: Method) -> bool:
    # TODO: Also inline if heuristics suggest it should be inlined.
    return method.inline


# Compilation process:
# * start with expression and a compilation context
# * compile to tree-form high level IR
# * do some optimizations, including inlining / intrinsic handling
# * flatten the tree into lower level IR
# * convert to bytecodes (or do native compilation... but that's a later project)

S = TypeVar("S")
T = TypeVar("T")


# Compiler for a specific method / quote body in a particular CompilationContext.
class Compiler:
    ir: TreeIRBlock
    # List of multimethods which, if changed, invalidate the compilation.
    multimethod_deps: list[MultiMethod]
    # Keep track of virtual register count for allocation purposes.
    num_virtual_regs: int
    # Keep track of label count for allocation purposes.
    num_labels: int

    basic_blocks: list[BasicIRBlock]
    entry_basic_block: Optional[BasicIRBlock]
    # Keep track of label count for allocation purposes.
    num_basic_blocks: int

    # Parameter types, or None if any type is allowed / there wasn't a parameter matcher.
    parameter_types: dict[SlotRegister, Optional[TypeValue]]

    lr_blocks: list[LRBlock]
    entry_lr_block: Optional[LRBlock]

    low_level_bytecode: list[LROp]

    # Internal use only.
    def __init__(self, ir):
        self.ir = ir
        self.multimethod_deps = []
        self.num_virtual_regs = 0
        self.num_labels = 0
        self.basic_blocks = []
        self.entry_basic_block = None
        self.num_basic_blocks = 0
        self.parameter_types = {}
        self.lr_blocks = []
        self.entry_lr_block = None
        self.low_level_bytecode = []

    # Add an IR operation, and return the destination register (for convenience).
    def add_ir_op(self, block: TreeIRBlock, op: IROp) -> Optional[Register]:
        block.ops.append(op)
        return op.dst

    # Allocate the next available virtual register.
    def allocate_virtual_reg(self) -> VirtualRegister:
        reg = VirtualRegister(self.num_virtual_regs)
        self.num_virtual_regs += 1
        return reg

    # Allocate the next available label.
    def allocate_label(self) -> Label:
        label = Label(id=self.num_labels, span=None)
        self.num_labels += 1
        return label

    # Allocate the next available basic block id.
    def allocate_basic_block(self) -> BasicIRBlock:
        block = BasicIRBlock(
            id=self.num_basic_blocks,
            ops=[],
            incoming=set(),
            outgoing=set(),
            dominators=set(),
            types=None,
        )
        self.basic_blocks.append(block)
        self.num_basic_blocks += 1
        return block

    # Finds which IR op in the given block (or its subblocks) writes to a given register.
    def find_writer(self, block: TreeIRBlock, reg: Register) -> IROp:
        def find_or_none(block, reg):
            for op in block.ops:
                if op.dst == reg:
                    return op
                for sub_block in op.sub_blocks:
                    writer = find_or_none(sub_block, reg)
                    if writer:
                        return writer
            return None

        writer = find_or_none(block, reg)
        if writer:
            return writer
        else:
            raise ValueError(f"Didn't find any op writing {reg} in input block")

    @staticmethod
    def compile_quote_method(method: Method) -> "Compiler":
        assert isinstance(method.body, QuoteMethodBody)
        quote = method.body

        compiler = Compiler(
            ir=TreeIRBlock(
                ops=[],
                ctxt=CompilationContext.stacked_compilation_context(
                    slots=[], base=quote.compiled_body.comp_ctxt
                ),
                inlining_stack=[],
            )
        )

        # TODO: offset slot registers (initial inputs to method) if there are already slot registers assigned?
        # e.g. if defining a method within another method.
        # method definitions outside of top level are probably totally broken right now...
        if quote.tail_recursive:
            span = quote.compiled_body.body.span

            entry_point = compiler.allocate_label()
            compiler.ir.ops.append(entry_point)

            # Includes default-receiver.
            argument_phi_ops = []
            for i, name in enumerate([default_receiver] + quote.param_names):
                op = PhiOp(
                    dst=compiler.allocate_virtual_reg(),
                    srcs=[(SlotRegister(i), None)],
                    force_keep=True,
                    span=span,
                )
                argument_phi_ops.append(op)
                compiler.ir.ops.append(op)
                compiler.ir.ctxt.slots.append((name, op.dst))

            compiler.ir.inlining_stack.append(
                InliningInfo(
                    body=quote,
                    allows_tail_recursion=True,
                    argument_phi_ops=argument_phi_ops,
                    entry_point=entry_point,
                )
            )
        else:
            compiler.ir.ctxt.slots.append((default_receiver, SlotRegister(0)))
            for i, name in enumerate(quote.param_names):
                compiler.ir.ctxt.slots.append((name, SlotRegister(i + 1)))

        # None = no restrictions / any type.
        compiler.parameter_types[SlotRegister(0)] = None
        assert len(method.param_matchers) == len(quote.param_names)
        for i, matcher in enumerate(method.param_matchers):
            slot = SlotRegister(i + 1)
            if isinstance(matcher, ParameterTypeMatcher):
                compiler.parameter_types[slot] = matcher.param_type
            else:
                # TODO: incorporate value matchers too (singleton type, or something?).
                compiler.parameter_types[slot] = None

        compiler.compile_body(quote.compiled_body.body)

        return compiler

    @staticmethod
    def compile_toplevel_expr(ctxt: Context, expr: Expr) -> "Compiler":
        compiler = Compiler(
            ir=TreeIRBlock(
                ops=[],
                ctxt=CompilationContext.stacked_compilation_context(slots=[], base=ctxt),
                inlining_stack=[],
            )
        )

        compiler.ir.ctxt.slots.append(
            (
                default_receiver,
                compiler.add_ir_op(
                    compiler.ir,
                    LiteralOp(
                        dst=compiler.allocate_virtual_reg(), value=NullValue(), span=expr.span
                    ),
                ),
            )
        )

        compiler.compile_body(expr)

        return compiler

    def compile_body(self, expr: Expr) -> None:
        try:
            print(colored("~~~~~~~~~ COMPILATION PROCESS ~~~~~~~~~~~~", "cyan"))

            print("input expression:", expr)
            print(colored("input compilation context (bottom -> top of stack):", "green"))
            stack_from_top = []
            ctxt = self.ir.ctxt
            while isinstance(ctxt, CompilationContext):
                stack_from_top.append(ctxt)
                ctxt = ctxt.base

            for ctxt in reversed(stack_from_top):
                print("  context:")
                for slot, value in ctxt.slots:
                    print(f"    {slot} -> {value}")

            print(colored("initial compilation to tree IR:", "red"))
            result_reg = self.compile_expr(
                self.ir, expr, tail_position=True, currently_inlining=False
            )
            if not result_reg:
                result_reg = self.add_ir_op(
                    self.ir,
                    LiteralOp(dst=self.allocate_virtual_reg(), value=NullValue(), span=expr.span),
                )
            self.add_ir_op(self.ir, ReturnOp(value=result_reg, span=expr.span))
            print_ir_block(self.ir, depth=1)

            any_change = True
            iteration = 0

            def optimize_tree(summary: str, fn):
                nonlocal any_change
                print(colored(f"{summary} (iteration={iteration}):", "red"))
                delta = fn()
                if delta:
                    any_change = True
                    print_ir_block(self.ir, depth=1)
                else:
                    print("    (no delta)")
                "a breakpoint right after printing latest state"

            while any_change:
                any_change = False
                iteration += 1

                # Inline multimethod dispatches to open up the method invocations.
                optimize_tree(
                    "inlining multimethod dispatches", self.tree_inline_multimethod_dispatches
                )
                # Inline method invocations to open up the method bodies and hopefully allow
                # closure inlining.
                optimize_tree(
                    "inlining methods",
                    lambda: self.tree_inline_methods(allow_quote_method_inlining=True),
                )
                # Try to move closures to their call sites.
                optimize_tree("propagating copies", self.tree_propagate_copies)
                optimize_tree("inlining closures", self.tree_inline_closures)

            # Tail recursion could have produced unnecessary phi nodes; now we can clean those up.
            optimize_tree("simplifying phi nodes post-inlining", self.tree_simplify_phi_nodes)
            # !! From this point on, no more inlining methods. !!
            # (We still can inline closures, though.)

            while any_change:
                any_change = False
                iteration += 1

                # Inline multimethod dispatches to open up the method invocations.
                optimize_tree(
                    "inlining multimethod dispatches", self.tree_inline_multimethod_dispatches
                )
                # Inline method invocations _except_ for quote methods.
                optimize_tree(
                    "inlining methods",
                    lambda: self.tree_inline_methods(allow_quote_method_inlining=False),
                )
                # Try to move closures to their call sites.
                optimize_tree("propagating copies", self.tree_propagate_copies)
                optimize_tree("inlining closures", self.tree_inline_closures)

            print(colored("converting to basic blocks:", "red"))
            self.convert_to_basic_blocks()
            self.compute_dominators()
            print_basic_blocks(self.basic_blocks)
            self.validate_cfg()

            def optimize_cfg(summary: str, fn):
                nonlocal any_change
                print(colored(f"{summary} (iteration={iteration}):", "red"))
                delta = fn()
                if delta:
                    any_change = True
                    print_basic_blocks(self.basic_blocks)
                else:
                    print("(no delta)")
                self.validate_cfg()
                "a breakpoint right after printing latest state"

            any_change = True
            while any_change:
                any_change = False
                iteration += 1
                optimize_cfg("simplifying control flow graph", self.simplify_control_flow_graph)
                optimize_cfg("eliminating dead/unreachable code", self.cfg_eliminate_dead_code)
                optimize_cfg("inferring types", self.cfg_infer_types)
                print_basic_blocks(self.basic_blocks)
                optimize_cfg("simplifying multimethod dispatches", self.cfg_simplify_dispatches)
                optimize_cfg("propagating copies", self.cfg_propagate_copies)
                for block in self.basic_blocks:
                    block.types = None

            print(colored("compiling to low level blocks:", "red"))
            self.compile_to_low_level_blocks()
            print_low_level_blocks(self.lr_blocks)
            # print(colored("linearizing to low level bytecode:", "red"))
            # self.compile_to_low_level_bytecode()
            # print_low_level_bytecode(self.low_level_bytecode)

            print(colored("~~~~~~~~~ COMPILATION PROCESS END ~~~~~~~~~~~~", "grey"))
        except Exception as e:
            print(f"Compilation error while compiling body: {expr}")
            print(f"Error: {e}")
            raise e

    def validate_cfg(self) -> None:
        # TODO: also do some global validation:
        # * single entry block
        # * actually SSA -- every reg assigned just once
        reachable_blocks = self.find_reachable_blocks()
        for block in self.basic_blocks:
            block.validate(is_reachable=(block in reachable_blocks))

    # Add IROps which evaluate the given expression.
    def compile_expr(
        self, block: TreeIRBlock, expr: Expr, tail_position: bool, currently_inlining: bool
    ) -> Optional[Register]:
        assert expr is not None

        # TODO: make this less hacky -- should be part of macro / AST rewrite system.
        tail_call = False
        if isinstance(expr, NAryMessageExpr) and [message.value for message in expr.messages] == [
            "TAIL-CALL"
        ]:
            tail_call = True
            if not tail_position:
                if currently_inlining:
                    # This is ok -- the tail-call got inlined somewhere where it is no longer in
                    # tail position. Inlining is stronger than tail-calling, anyway.
                    tail_call = False
                else:
                    raise ValueError("TAIL-CALL: must be invoked in tail position")

            if expr.target is not None:
                raise ValueError("TAIL-CALL: requires no receiver")
            call = expr.args[0]
            while isinstance(call, ParenExpr):
                call = call.inner
            if not isinstance(
                call, (UnaryOpExpr, BinaryOpExpr, NameExpr, UnaryMessageExpr, NAryMessageExpr)
            ):
                raise ValueError(
                    "TAIL-CALL: must be applied to a unary/binary op or a unary/n-ary message"
                )
            expr = call

        # Only the first of the args is actually optional.
        def compile_invocation(
            message: str, args: list[Optional[Expr]], tail_call: bool, span: SourceSpan
        ) -> Optional[Register]:
            assert len(args) > 0
            assert all(arg is not None for arg in args[1:])

            slot = block.ctxt.lookup(message)
            if not slot:
                raise ValueError(f"Could not compile: unknown slot '{message}'.")
            assert isinstance(slot, (Register, Value, MultiMethod, CompileTimeHandler)), slot

            if isinstance(slot, Register):
                receiver, args = args[0], args[1:]
                if receiver or args:
                    raise ValueError(
                        f"Could not compile: local slot does not need receiver or arguments: {message}."
                    )
                return slot
            if isinstance(slot, CompileTimeHandler):
                return slot.handler(self, block, span, tail_call, *args)
            elif isinstance(slot, Value):
                receiver, args = args[0], args[1:]
                if receiver or args:
                    raise ValueError(
                        f"Could not compile: slot is not a method, so does not need receiver or arguments: {message}."
                    )
                # TODO: could do a similar thing to multimethods, where we add a lease to the slot and if it changes
                # it forces recompilation. Might cause too many invalidations though.
                return self.add_ir_op(
                    block,
                    SlotLookupOp(dst=self.allocate_virtual_reg(), slot_name=message, span=span),
                )
            elif isinstance(slot, MultiMethod):
                arg_regs = []
                for arg in args:
                    if arg is None:
                        default_receiver_reg = block.ctxt.lookup(default_receiver)
                        assert isinstance(default_receiver_reg, Register)
                        arg_reg = default_receiver_reg
                    else:
                        arg_reg = self.compile_expr(
                            block, arg, tail_position=False, currently_inlining=currently_inlining
                        )
                    arg_regs.append(arg_reg)

                # Even if a tail call, we must add a destination register to be used in case of dispatch failure.
                invocation = InvokeMultimethodOp(
                    dst=self.allocate_virtual_reg(),
                    multimethod=slot,
                    call_args=arg_regs,
                    tail_call=tail_call,
                    tail_position=tail_position,
                    span=span,
                )
                return self.add_ir_op(block, invocation)
            else:
                raise AssertionError(f"forgot a slot type? {type(slot)}: {slot}")

        if isinstance(expr, UnaryOpExpr):
            return compile_invocation(
                expr.op.value, [expr.arg], tail_call=tail_call, span=expr.span
            )
        elif isinstance(expr, BinaryOpExpr):
            return compile_invocation(
                expr.op.value + ":", [expr.left, expr.right], tail_call=tail_call, span=expr.span
            )
        elif isinstance(expr, NameExpr):
            return compile_invocation(expr.name.value, [None], tail_call=tail_call, span=expr.span)
        elif isinstance(expr, LiteralExpr):
            literal = expr.literal
            if literal._type == TokenType.SYMBOL:
                value = SymbolValue(literal.value)
            elif literal._type == TokenType.NUMBER:
                value = NumberValue(literal.value)
            elif literal._type == TokenType.STRING:
                value = StringValue(literal.value)
            else:
                raise AssertionError(f"Forgot a literal token type! {literal._type} ({literal})")
            return self.add_ir_op(
                block, LiteralOp(dst=self.allocate_virtual_reg(), value=value, span=expr.span)
            )
        elif isinstance(expr, UnaryMessageExpr):
            return compile_invocation(
                expr.message.value, [expr.target], tail_call=tail_call, span=expr.span
            )
        elif isinstance(expr, NAryMessageExpr):
            assert len(expr.messages) == len(expr.args)
            message = "".join(message.value + ":" for message in expr.messages)
            return compile_invocation(
                message, [expr.target] + expr.args, tail_call=tail_call, span=expr.span
            )
        elif isinstance(expr, ParenExpr):
            return self.compile_expr(block, expr.inner, tail_position, currently_inlining)
        elif isinstance(expr, QuoteExpr):
            # TODO: this is where `it` default param needs to be added (or not).
            if not expr.parameters:
                param_names = ["it"]
            else:
                param_names = expr.parameters
            # TODO: what's right here?
            body_comp_ctxt = CompilationContext.stacked_compilation_context(
                slots=[], base=block.ctxt
            )
            # body_comp_ctxt = CompilationContext.stacked_compilation_context(
            #    slots=[(default_receiver, SlotRegister(0))]
            #    + [(param, SlotRegister(i + 1)) for i, param in enumerate(param_names)],
            #    base=block.ctxt,
            # )
            quote = QuoteValue(
                parameters=expr.parameters,
                compiled_body=CompiledBody(
                    expr.body, bytecode=None, comp_ctxt=body_comp_ctxt, method=None
                ),
                context=None,  # will be filled in later during evaluation
                span=expr.span,
            )
            return self.add_ir_op(
                block, ClosureOp(dst=self.allocate_virtual_reg(), quote=quote, span=expr.span)
            )
        elif isinstance(expr, DataExpr):
            component_regs = []
            for component in expr.components:
                reg = self.compile_expr(
                    block, component, tail_position=False, currently_inlining=currently_inlining
                )
                # Each component should produce a result into a register... no tail calls possible here.
                assert reg
                component_regs.append(reg)
            return self.add_ir_op(
                block,
                VectorOp(
                    dst=self.allocate_virtual_reg(), components=component_regs, span=expr.span
                ),
            )
        elif isinstance(expr, SequenceExpr):
            assert expr.sequence != []
            last_output = None
            for i, part in enumerate(expr.sequence):
                part_is_tail_position = tail_position and (i == len(expr.sequence) - 1)
                last_output = self.compile_expr(
                    block, part, part_is_tail_position, currently_inlining=currently_inlining
                )
            return last_output
        elif isinstance(expr, TupleExpr):
            component_regs = []
            for component in expr.components:
                reg = self.compile_expr(
                    block, component, tail_position=False, currently_inlining=currently_inlining
                )
                # Each component should produce a result into a register... no tail calls possible here.
                assert reg
                component_regs.append(reg)
            return self.add_ir_op(
                block, TupleOp(dst=self.allocate_virtual_reg(), components=component_regs)
            )
        else:
            raise AssertionError(f"Forgot an expression type! {type(expr)}")

    def tree_inline_multimethod_dispatches(self) -> bool:
        def process_block(block: TreeIRBlock) -> bool:
            old_ops = list(block.ops)
            block.ops = []

            any_change = False

            for op in old_ops:
                if not isinstance(op, InvokeMultimethodOp):
                    block.ops.append(op)
                    for sub_block in op.sub_blocks:
                        if process_block(sub_block):
                            any_change = True
                    continue

                multimethod = op.multimethod
                if not should_inline_multimethod_dispatch(multimethod):
                    block.ops.append(op)
                    continue

                any_change = True

                if multimethod not in self.multimethod_deps:
                    self.multimethod_deps.append(multimethod)

                dispatch = []
                method_results = []
                sub_blocks = []
                for method in multimethod.methods:
                    method_block = TreeIRBlock(
                        ops=[], ctxt=block.ctxt, inlining_stack=block.inlining_stack
                    )
                    dispatch.append((method, method_block))
                    sub_blocks.append(method_block)

                    dst = self.allocate_virtual_reg()
                    method_results.append((dst, None))
                    start_label = self.allocate_label()
                    method_block.ops.append(start_label)
                    method_block.ops.append(
                        InvokeMethodOp(
                            dst=dst,
                            method=method,
                            call_args=op.call_args,
                            tail_call=op.tail_call,
                            tail_position=op.tail_position,
                            method_start_label=start_label,
                            span=op.span,
                        )
                    )
                    # Note that even if a tail call, code after may still be reachable, for instance
                    # if the method invocation gets inlined (and so is no longer a tail call).

                # Add more sub-blocks for the 'dispatch failed' cases.
                no_matching_method = TreeIRBlock(
                    ops=[], ctxt=block.ctxt, inlining_stack=block.inlining_stack
                )
                no_matching_method_signal_result = self.allocate_virtual_reg()
                no_matching_method_signal_op = SignalOp(
                    dst=no_matching_method_signal_result,
                    condition_name="no-matching-method",
                    signal_args=op.call_args,
                    span=op.span,
                )
                no_matching_method.ops.append(no_matching_method_signal_op)
                sub_blocks.append(no_matching_method)

                # We can do a simple optimization here; ambiguous-method-resolution is impossible
                # if there are only zero or one dispatch options. (I expect this is a pretty common case.)
                if len(multimethod.methods) != 1:
                    ambiguous_method_resolution = TreeIRBlock(
                        ops=[], ctxt=block.ctxt, inlining_stack=block.inlining_stack
                    )
                    ambiguous_method_resolution_signal_result = self.allocate_virtual_reg()
                    ambiguous_method_resolution_signal_op = SignalOp(
                        dst=ambiguous_method_resolution_signal_result,
                        condition_name="ambiguous-method-resolution",
                        signal_args=op.call_args,
                        span=op.span,
                    )
                    ambiguous_method_resolution.ops.append(no_matching_method_signal_op)
                    sub_blocks.append(ambiguous_method_resolution)
                else:
                    ambiguous_method_resolution = None
                    ambiguous_method_resolution_signal_result = None
                    ambiguous_method_resolution_signal_op = None

                block.ops.append(
                    MultimethodDispatchOp(
                        dst=None,
                        slot_name=multimethod.name,
                        dispatch_args=op.call_args,
                        dispatch=dispatch,
                        no_matching_method=no_matching_method,
                        ambiguous_method_resolution=ambiguous_method_resolution,
                        sub_blocks=sub_blocks,
                        span=op.span,
                    )
                )

                if op.dst:
                    # Re-use the old invoke op's destination register so that downstream consumers
                    # still get a value available in this register.
                    block.ops.append(
                        PhiOp(
                            dst=op.dst,
                            srcs=method_results
                            + [(no_matching_method_signal_result, no_matching_method_signal_op)]
                            + (
                                [
                                    (
                                        ambiguous_method_resolution_signal_result,
                                        ambiguous_method_resolution_signal_op,
                                    )
                                ]
                                if ambiguous_method_resolution_signal_result
                                else []
                            ),
                            span=op.span,
                        )
                    )

            return any_change

        return process_block(self.ir)

    def tree_inline_methods(self, allow_quote_method_inlining: bool):
        def process_block(block: TreeIRBlock) -> bool:
            old_ops = list(block.ops)
            block.ops = []

            any_change = False

            for op in old_ops:
                if not isinstance(op, InvokeMethodOp):
                    block.ops.append(op)
                    for sub_block in op.sub_blocks:
                        if process_block(sub_block):
                            any_change = True
                    continue

                any_change = True

                method = op.method
                if isinstance(method.body, QuoteMethodBody):
                    if any(method.body == inlined.body for inlined in block.inlining_stack):
                        inlined = next(
                            info for info in block.inlining_stack if method.body == info.body
                        )
                    else:
                        inlined = None
                    # Stop inlining if we have already inlined the same method, but it doesn't support tail recursion.
                    # If we did try to inline, we would either have to jump back to the entry point (which doesn't exist,
                    # since we didn't compile the original inlining assuming there would be tail recursion), or actually
                    # inline again, which would lead to infinite compilation recursion.
                    if (
                        allow_quote_method_inlining
                        and (not inlined or inlined.allows_tail_recursion)
                        and should_inline_quote_method(method)
                    ):
                        # If already in the inlining stack, then we can handle this as a jump to where we
                        # already inlined this method body.
                        if inlined:
                            jump_op = JumpOp(target=inlined.entry_point, span=op.span)

                            # TODO: also need to have phi ops for _locals_, not just arguments... otherwise
                            # local mutation with tail recursion optimization effectively forgets the mutation.
                            assert len(inlined.argument_phi_ops) == 1 + len(method.body.param_names)
                            for phi_op in inlined.argument_phi_ops:
                                assert isinstance(phi_op, PhiOp)
                                assert phi_op.force_keep
                            default_receiver_phi_op, arg_phi_ops = (
                                inlined.argument_phi_ops[0],
                                inlined.argument_phi_ops[1:],
                            )

                            # Copy arguments into the right slots.
                            # The default receiver is just <null>.
                            default_receiver_reg = self.add_ir_op(
                                block,
                                LiteralOp(
                                    dst=self.allocate_virtual_reg(),
                                    value=NullValue(),
                                    span=op.span,
                                ),
                            )
                            default_receiver_phi_op.srcs.append((default_receiver_reg, jump_op))

                            assert len(op.call_args) == len(method.body.param_names)
                            for i, param_name in enumerate(method.body.param_names):
                                arg_phi_ops[i].srcs.append((op.call_args[i], jump_op))

                            self.add_ir_op(block, JumpOp(target=inlined.entry_point, span=op.span))
                            self.add_ir_op(block, UnreachableOp(dst=op.dst, span=op.span))
                        else:
                            # Set up arguments.
                            # The default receiver is just <null>.
                            # TODO: this is super hacky, make it not hacky.
                            i = block.ops.index(op.method_start_label)
                            block.ops.insert(
                                i,
                                LiteralOp(
                                    dst=self.allocate_virtual_reg(),
                                    value=NullValue(),
                                    span=op.span,
                                ),
                            )
                            default_receiver_reg = block.ops[i].dst
                            if method.body.tail_recursive:
                                default_receiver_phi_op = PhiOp(
                                    dst=self.allocate_virtual_reg(),
                                    srcs=[(default_receiver_reg, None)],
                                    force_keep=True,
                                    span=op.span,
                                )

                            assert len(op.call_args) == len(method.body.param_names)
                            if method.body.tail_recursive:
                                argument_phi_ops = []
                                for i, arg_reg in enumerate(op.call_args):
                                    argument_phi_ops.append(
                                        PhiOp(
                                            dst=self.allocate_virtual_reg(),
                                            srcs=[(arg_reg, None)],
                                            force_keep=True,
                                            span=op.span,
                                        )
                                    )

                            if method.body.tail_recursive:
                                block.ops.append(default_receiver_phi_op)
                                block.ops.extend(argument_phi_ops)

                            # "Push" the inline context.
                            if method.body.tail_recursive:
                                inline_ctxt = CompilationContext.stacked_compilation_context(
                                    slots=[(default_receiver, default_receiver_phi_op.dst)]
                                    + [
                                        (param, argument_phi_ops[i].dst)
                                        for i, param in enumerate(method.body.param_names)
                                    ],
                                    base=block.ctxt,
                                )
                            else:
                                inline_ctxt = CompilationContext.stacked_compilation_context(
                                    slots=[(default_receiver, default_receiver_reg)]
                                    + [
                                        (param, op.call_args[i])
                                        for i, param in enumerate(method.body.param_names)
                                    ],
                                    base=block.ctxt,
                                )

                            # Indicate that we are currently inlining the method, so if we find recursive evaluation
                            # we know we can just jump back to the starting point of the multimethod.
                            inline_block = TreeIRBlock(
                                ops=[],
                                ctxt=inline_ctxt,
                                inlining_stack=block.inlining_stack
                                + [
                                    (
                                        InliningInfo(
                                            body=method.body,
                                            allows_tail_recursion=True,
                                            argument_phi_ops=[default_receiver_phi_op]
                                            + argument_phi_ops,
                                            entry_point=op.method_start_label,
                                        )
                                        if method.body.tail_recursive
                                        else InliningInfo(
                                            body=method.body,
                                            allows_tail_recursion=False,
                                            argument_phi_ops=None,
                                            entry_point=None,
                                        )
                                    )
                                ],
                            )

                            # Finally we are ready to inline the body.
                            result = self.compile_expr(
                                inline_block,
                                method.body.compiled_body.body,
                                tail_position=op.tail_position,
                                currently_inlining=True,
                            )

                            # Add the inline block!
                            self.add_ir_op(
                                block, InlineBlockOp(sub_blocks=[inline_block], span=None)
                            )

                            if result:
                                # Copy the result back to where the old invocation operation wrote the result.
                                self.add_ir_op(block, CopyOp(dst=op.dst, src=result, span=op.span))
                    else:
                        # Just invoke the quote body. We can reuse most of the op fields; this is just replacing
                        # an InvokeMethodOp with a more specific InvokeQuoteOp.
                        self.add_ir_op(
                            block,
                            InvokeQuoteOp(
                                dst=op.dst,
                                quote=method.body,
                                call_args=op.call_args,
                                tail_call=op.tail_call,
                                tail_position=op.tail_position,
                                span=op.span,
                            ),
                        )
                elif isinstance(method.body, IntrinsicMethodBody):
                    if method.body.compile_inline is not None:
                        # Ask the intrinsic to compile inline!
                        result = method.body.compile_inline(
                            self, block, op.call_args, op.tail_call, op.tail_position, op.span
                        )
                        assert result is None or isinstance(result, Register)
                        if op.dst and result:
                            # Make sure to write to the old-invocation output register.
                            self.add_ir_op(block, CopyOp(dst=op.dst, src=result, span=op.span))
                        # TODO: what if op.dst but not result? delete the old output register somehow?
                        # should just be handled by dead code elimination later on, I s'pose...
                    else:
                        # Just invoke the intrinsic. We can reuse most of the op fields; this is just replacing
                        # an InvokeMethodOp with a more specific InvokeIntrinsicOp.
                        self.add_ir_op(
                            block,
                            InvokeIntrinsicOp(
                                dst=op.dst,
                                intrinsic=method.body.handler,
                                call_args=op.call_args,
                                tail_call=op.tail_call,
                                tail_position=op.tail_position,
                                span=op.span,
                            ),
                        )
                elif isinstance(method.body, NativeMethodBody):
                    # TODO: inline compile natives?
                    # Just invoke the native. We can reuse most of the op fields; this is just replacing
                    # an InvokeMethodOp with a more specific InvokeNativeOp.
                    self.add_ir_op(
                        block,
                        InvokeNativeOp(
                            dst=op.dst,
                            native=method.body.handler,
                            return_type=method.return_type,
                            call_args=op.call_args,
                            tail_call=op.tail_call,
                            tail_position=op.tail_position,
                            span=op.span,
                        ),
                    )
                else:
                    raise AssertionError(f"forgot a method body type: {type(method.body)}")

            return any_change

        return process_block(self.ir)

    def tree_propagate_copies(self) -> bool:
        # Propagate CopyOp assignments forward, even into sub-blocks, until we run into any
        # walls: any JumpOp.
        # We can also merge remappings modified variously by different branches (sub-blocks)
        # of an op.
        # TODO: do we actually need to clear out remappings on JumpOp? maybe not...

        # Remapping is from destination to source in various assignments.
        # Returns a new remapping, and a flag saying if any ops were modified.
        def process_block(
            block: TreeIRBlock, remapping: dict[Register, Register]
        ) -> Tuple[dict, bool]:
            remapping = dict(remapping)

            any_change = False

            for op in block.ops:
                remapping_per_subblock = []
                for sub_block in op.sub_blocks:
                    subblock_remapping, changed = process_block(sub_block, remapping)
                    if changed:
                        any_change = True
                    remapping_per_subblock.append(subblock_remapping)

                # If there were any branches in the first place, then merge them and use that as
                # our new remapping.
                if remapping_per_subblock:
                    for merger in remapping_per_subblock:
                        for k, v in merger.items():
                            if k not in remapping:
                                remapping[k] = v
                            if remapping[k] != v:
                                del remapping[k]

                def remap(reg: Register) -> Register:
                    nonlocal any_change
                    remapped = remapping.get(reg, reg)
                    if remapped != reg:
                        any_change = True
                    return remapped

                if isinstance(op, InlineBlockOp):
                    # We already got this as part of the sub-block copy propagation.
                    pass
                elif isinstance(op, Label):
                    pass
                elif isinstance(op, JumpOp):
                    # TODO: we don't have to forget all the remappings when hitting a JumpOp, just
                    # mappings that were established after the JumpOp target label. (I think...)
                    remapping = {}
                elif isinstance(op, UnreachableOp):
                    pass
                elif isinstance(op, ReturnOp):
                    op.value = remap(op.value)
                elif isinstance(op, CopyOp):
                    op.src = remap(op.src)
                    if op.dst not in remapping:
                        remapping[op.dst] = op.src
                elif isinstance(op, PhiOp):
                    op.srcs = [(remap(reg), block) for reg, block in op.srcs]
                elif isinstance(op, LiteralOp):
                    pass
                elif isinstance(op, InvokeRegisterOp):
                    op.callable = remap(op.callable)
                    op.call_args = [remap(arg) for arg in op.call_args]
                elif isinstance(op, InvokeMultimethodOp):
                    op.call_args = [remap(arg) for arg in op.call_args]
                elif isinstance(op, InvokeMethodOp):
                    op.call_args = [remap(arg) for arg in op.call_args]
                elif isinstance(op, InvokeQuoteOp):
                    op.call_args = [remap(arg) for arg in op.call_args]
                elif isinstance(op, InvokeIntrinsicOp):
                    op.call_args = [remap(arg) for arg in op.call_args]
                elif isinstance(op, InvokeNativeOp):
                    op.call_args = [remap(arg) for arg in op.call_args]
                elif isinstance(op, ClosureOp):
                    pass
                elif isinstance(op, SlotLookupOp):
                    pass
                elif isinstance(op, VectorOp):
                    op.components = [remap(comp) for comp in op.components]
                elif isinstance(op, TupleOp):
                    op.components = [remap(comp) for comp in op.components]
                elif isinstance(op, MultimethodDispatchOp):
                    op.dispatch_args = [remap(arg) for arg in op.dispatch_args]
                elif isinstance(op, SignalOp):
                    op.signal_args = [remap(arg) for arg in op.signal_args]
                elif isinstance(op, IfElseOp):
                    op.condition = remap(op.condition)
                else:
                    raise AssertionError(f"forgot an IROp: {type(op)}")

            return remapping, any_change

        _, any_change = process_block(self.ir, remapping={})
        return any_change

    def tree_inline_closures(self) -> bool:
        # Propagates ClosureOp assignments directly forward to InvokeRegisterOp calls,
        # and inlines these closures at their call sites.

        def process_block(block: TreeIRBlock, closures: dict[Register, QuoteValue]) -> bool:
            closures = dict(closures)
            old_ops = block.ops
            block.ops = []

            any_change = False

            for op in old_ops:
                for sub_block in op.sub_blocks:
                    if process_block(sub_block, closures):
                        any_change = True

                if isinstance(op, InlineBlockOp):
                    # Nothing else to do; we already got this as part of the sub-block processing.
                    block.ops.append(op)
                elif isinstance(op, Label):
                    block.ops.append(op)
                elif isinstance(op, JumpOp):
                    # TODO: do we actually need to clear out mappings here? Maybe not...
                    closures = {}
                    block.ops.append(op)
                elif isinstance(op, UnreachableOp):
                    block.ops.append(op)
                elif isinstance(op, ReturnOp):
                    block.ops.append(op)
                elif isinstance(op, CopyOp):
                    block.ops.append(op)
                elif isinstance(op, PhiOp):
                    block.ops.append(op)
                elif isinstance(op, LiteralOp):
                    block.ops.append(op)
                elif isinstance(op, InvokeRegisterOp):
                    if op.callable in closures:
                        # Yay, we can inline!
                        # TODO: see about deduplicating with inline_methods().

                        # The "calling convention" here matches that of call_impl() in the interpreter:
                        # #parameters | #arguments  |  parameters   | default-receiver  |  passed values to parameters
                        # ------------+-------------+----------------------------------------------------------------------------------------
                        #      0      |     0       |   add "it"    |      null         |  null
                        #      0      |     1       |   add "it"    |     args[0]       |  args[0]
                        #      1      |     1       |      -        |     args[0]       |  args[0]
                        #        <otherwise>        |      -        |      null         |  args[0], args[1], ...

                        quote = closures[op.callable]

                        if not (len(quote.parameters) == 0 and len(op.call_args) <= 1) and len(
                            quote.parameters
                        ) != len(op.call_args):
                            raise ValueError(
                                f"quote has {len(quote.parameters)} parameter(s), but is being provided {len(op.call_args)} argument(s)"
                            )

                        # Set up arguments.
                        if len(op.call_args) == 1:
                            default_receiver_reg = op.call_args[0]
                        else:
                            default_receiver_reg = self.add_ir_op(
                                block,
                                LiteralOp(
                                    dst=self.allocate_virtual_reg(),
                                    value=NullValue(),
                                    span=op.span,
                                ),
                            )

                        if not quote.parameters and not op.call_args:
                            arg_reg = self.add_ir_op(
                                block,
                                LiteralOp(
                                    dst=self.allocate_virtual_reg(),
                                    value=NullValue(),
                                    span=op.span,
                                ),
                            )
                            arg_regs = [arg_reg]
                        else:
                            arg_regs = op.call_args

                        # "Push" the inline context.
                        inline_ctxt = CompilationContext.stacked_compilation_context(
                            slots=[(default_receiver, default_receiver_reg)]
                            + [
                                (param, arg_regs[i])
                                for i, param in enumerate(quote.parameters or ["it"])
                            ],
                            # TODO: block.ctxt or quote.ctxt?
                            base=quote.compiled_body.comp_ctxt,
                        )

                        # No new quote-method-bodies inlined, just a closure quote. Not tracked on the inlining stack,
                        # since these are first class values (unlike multimethods / multimethod invocation).
                        inline_block = TreeIRBlock(
                            ops=[], ctxt=inline_ctxt, inlining_stack=block.inlining_stack
                        )

                        # Finally we are ready to inline the body.
                        result = self.compile_expr(
                            inline_block,
                            quote.compiled_body.body,
                            tail_position=op.tail_position,
                            currently_inlining=True,
                        )

                        # Add the inline block!
                        self.add_ir_op(block, InlineBlockOp(sub_blocks=[inline_block], span=None))

                        if op.dst and result:
                            # Copy the result back to where the old invocation operation wrote the result.
                            self.add_ir_op(block, CopyOp(dst=op.dst, src=result, span=op.span))

                        any_change = True
                    else:
                        block.ops.append(op)
                elif isinstance(op, InvokeMultimethodOp):
                    block.ops.append(op)
                elif isinstance(op, InvokeMethodOp):
                    block.ops.append(op)
                elif isinstance(op, InvokeQuoteOp):
                    block.ops.append(op)
                elif isinstance(op, InvokeIntrinsicOp):
                    block.ops.append(op)
                elif isinstance(op, InvokeNativeOp):
                    block.ops.append(op)
                elif isinstance(op, ClosureOp):
                    assert op.dst not in closures
                    closures[op.dst] = op.quote
                    block.ops.append(op)
                elif isinstance(op, SlotLookupOp):
                    block.ops.append(op)
                elif isinstance(op, VectorOp):
                    block.ops.append(op)
                elif isinstance(op, TupleOp):
                    block.ops.append(op)
                elif isinstance(op, MultimethodDispatchOp):
                    block.ops.append(op)
                elif isinstance(op, SignalOp):
                    block.ops.append(op)
                elif isinstance(op, IfElseOp):
                    block.ops.append(op)
                else:
                    raise AssertionError(f"forgot an IROp: {type(op)}")

            return any_change

        return process_block(self.ir, closures={})

    def tree_simplify_phi_nodes(self) -> bool:
        def process_block(block: TreeIRBlock) -> bool:
            old_ops = block.ops
            block.ops = []

            any_change = False

            for op in old_ops:
                for sub_block in op.sub_blocks:
                    if process_block(sub_block):
                        any_change = True

                if not isinstance(op, PhiOp):
                    block.ops.append(op)
                    continue

                op.force_keep = False
                # TODO: this seems a bit smelly, and `swap:` example shows this isn't necessarily
                # the right approach to take... rework this (and inline-method argument re-writing)
                # and get `swap:` working.
                if any(reg == op.dst for reg, _ in op.srcs):
                    any_change = True
                    op.srcs = [(reg, block) for reg, block in op.srcs if reg != op.dst]
                assert op.srcs
                if len(op.srcs) == 1:
                    # Might as well convert to a copy!
                    reg, _ = op.srcs[0]
                    block.ops.append(CopyOp(dst=op.dst, src=reg, span=op.span))
                    any_change = True
                else:
                    block.ops.append(op)

            return any_change

        return process_block(self.ir)

    def convert_to_basic_blocks(self) -> bool:
        # Walks the tree block, converting instructions over to the provided basic block until reaching
        # any nonlinear control flow (at which point this creates new descendant basic blocks and proceeds).
        # Also is given (and updates) a global label to basic-block mapping, where the basic block begins just after the label.
        # Furthermore, is given (and updates) a global tree-IR-op to basic-block mapping, indicating which basic block holds
        #    whatever the tree IR op was converted into. (This only tracks jump ops and ops which produce outputs, though;
        #    for instance it should not track any entry for an if/else op.)
        # Returns the basic block hosting the tail end of ops from the original tree block.
        def process_tree_block(
            tree_block: TreeIRBlock,
            basic_block: BasicIRBlock,
            label_to_basic_block: dict[Label, BasicIRBlock],
            tree_ir_op_to_basic_block: list[Tuple[IROp, BasicIRBlock]],
        ) -> BasicIRBlock:
            current_block = basic_block
            for op in tree_block.ops:
                if isinstance(op, InlineBlockOp):
                    assert len(op.sub_blocks) == 1
                    current_block = process_tree_block(
                        op.sub_blocks[0],
                        current_block,
                        label_to_basic_block=label_to_basic_block,
                        tree_ir_op_to_basic_block=tree_ir_op_to_basic_block,
                    )
                elif isinstance(op, Label):
                    # Split at labels, since jumps may go to here.
                    assert op not in label_to_basic_block
                    new_block = self.allocate_basic_block()
                    label_to_basic_block[op] = new_block
                    basic_block.add_op(BasicBlockJumpOp(target=new_block, span=op.span))
                    basic_block.link_outgoing(new_block)
                    current_block = new_block
                elif isinstance(op, JumpOp):
                    tree_ir_op_to_basic_block.append((op, current_block))
                    # This should be fine to assume, since all jumps are backward. (Forward jumps are handled
                    # instead with specialised ops like IfElseOp.)
                    assert op.target in label_to_basic_block
                    target = label_to_basic_block[op.target]
                    current_block.add_op(BasicBlockJumpOp(target=target, span=op.span))
                    current_block.link_outgoing(target)
                    # Next instruction goes in a new basic block. (It will be unreachable!)
                    current_block = self.allocate_basic_block()
                elif isinstance(op, UnreachableOp):
                    tree_ir_op_to_basic_block.append((op, current_block))
                    current_block.add_op(op)
                elif isinstance(op, ReturnOp):
                    tree_ir_op_to_basic_block.append((op, current_block))
                    current_block.add_op(op)
                    # Next instruction goes in a new basic block. (It will be unreachable!)
                    current_block = self.allocate_basic_block()
                elif isinstance(op, PhiOp):
                    tree_ir_op_to_basic_block.append((op, current_block))
                    # The source blocks are remapped in a second pass.
                    current_block.add_op(
                        TransitionalPhiOp(
                            dst=op.dst,
                            srcs=[(reg, src_tree_block) for reg, src_tree_block in op.srcs],
                            span=op.span,
                        )
                    )
                elif isinstance(
                    op,
                    (
                        CopyOp,
                        LiteralOp,
                        BaseInvokeOp,
                        ClosureOp,
                        SlotLookupOp,
                        VectorOp,
                        TupleOp,
                    ),
                ):
                    tree_ir_op_to_basic_block.append((op, current_block))
                    current_block.add_op(op)
                elif isinstance(op, MultimethodDispatchOp):
                    dispatch = [(method, self.allocate_basic_block()) for method, _ in op.dispatch]
                    no_matching_method = (
                        self.allocate_basic_block() if op.no_matching_method else None
                    )
                    ambiguous_method_resolution = (
                        self.allocate_basic_block() if op.ambiguous_method_resolution else None
                    )
                    after_block = self.allocate_basic_block()

                    current_block.add_op(
                        BasicBlockMultimethodDispatchOp(
                            slot_name=op.slot_name,
                            dispatch_args=op.dispatch_args,
                            dispatch=dispatch,
                            no_matching_method=no_matching_method,
                            ambiguous_method_resolution=ambiguous_method_resolution,
                            span=op.span,
                        )
                    )
                    for _, method_block in dispatch:
                        current_block.link_outgoing(method_block)
                    if no_matching_method:
                        current_block.link_outgoing(no_matching_method)
                    if ambiguous_method_resolution:
                        current_block.link_outgoing(ambiguous_method_resolution)

                    for i in range(len(dispatch)):
                        _, method_tree_block = op.dispatch[i]
                        _, method_basic_block = dispatch[i]
                        method_basic_block = process_tree_block(
                            method_tree_block,
                            method_basic_block,
                            label_to_basic_block=label_to_basic_block,
                            tree_ir_op_to_basic_block=tree_ir_op_to_basic_block,
                        )
                        if method_basic_block.falls_through():
                            method_basic_block.add_op(
                                BasicBlockJumpOp(target=after_block, span=op.span)
                            )
                            method_basic_block.link_outgoing(after_block)

                    if op.no_matching_method:
                        no_matching_method = process_tree_block(
                            op.no_matching_method,
                            no_matching_method,
                            label_to_basic_block=label_to_basic_block,
                            tree_ir_op_to_basic_block=tree_ir_op_to_basic_block,
                        )
                        if no_matching_method.falls_through():
                            no_matching_method.add_op(
                                BasicBlockJumpOp(target=after_block, span=op.span)
                            )
                            no_matching_method.link_outgoing(after_block)

                    if op.ambiguous_method_resolution:
                        ambiguous_method_resolution = process_tree_block(
                            op.ambiguous_method_resolution,
                            ambiguous_method_resolution,
                            label_to_basic_block=label_to_basic_block,
                            tree_ir_op_to_basic_block=tree_ir_op_to_basic_block,
                        )
                        if ambiguous_method_resolution.falls_through():
                            ambiguous_method_resolution.add_op(
                                BasicBlockJumpOp(target=after_block, span=op.span)
                            )
                            ambiguous_method_resolution.link_outgoing(after_block)

                    current_block = after_block
                elif isinstance(op, SignalOp):
                    tree_ir_op_to_basic_block.append((op, current_block))
                    current_block.add_op(op)
                elif isinstance(op, IfElseOp):
                    true_block = self.allocate_basic_block()
                    false_block = self.allocate_basic_block()
                    after_block = self.allocate_basic_block()

                    current_block.add_op(
                        BasicBlockIfElseOp(
                            condition=op.condition,
                            true_block=true_block,
                            false_block=false_block,
                            span=op.span,
                        )
                    )
                    current_block.link_outgoing(true_block)
                    current_block.link_outgoing(false_block)

                    true_block = process_tree_block(
                        op.true_block,
                        true_block,
                        label_to_basic_block=label_to_basic_block,
                        tree_ir_op_to_basic_block=tree_ir_op_to_basic_block,
                    )
                    if true_block.falls_through():
                        true_block.add_op(BasicBlockJumpOp(target=after_block, span=op.span))
                        true_block.link_outgoing(after_block)

                    false_block = process_tree_block(
                        op.false_block,
                        false_block,
                        label_to_basic_block=label_to_basic_block,
                        tree_ir_op_to_basic_block=tree_ir_op_to_basic_block,
                    )
                    if false_block.falls_through():
                        false_block.add_op(BasicBlockJumpOp(target=after_block, span=op.span))
                        false_block.link_outgoing(after_block)

                    current_block = after_block
                else:
                    raise AssertionError(f"forgot an IROp: {type(op)}")
            return current_block

        assert not self.entry_basic_block
        self.entry_basic_block = self.allocate_basic_block()
        tree_ir_op_to_basic_block = []
        output_block = process_tree_block(
            self.ir,
            self.entry_basic_block,
            label_to_basic_block={},
            tree_ir_op_to_basic_block=tree_ir_op_to_basic_block,
        )
        if not output_block.ops:
            self.basic_blocks.remove(output_block)

        def lookup_tree_ir_op(op):
            for tree_op, block in tree_ir_op_to_basic_block:
                if tree_op == op:
                    return block
            raise AssertionError(
                f"didn't find {op} in the tree_ir_op_to_basic_block association list"
            )

        self.compute_dominators()

        # Finds the immediate predecessor of `dst` which `src` dominates, and for which every path from `src` to `dst`
        # passes through this predecessor before passing through `dst`. There should be exactly one.
        def source_block_to_predecessor(src: BasicIRBlock, dst: BasicIRBlock) -> BasicIRBlock:
            def is_path_blocker(src: BasicIRBlock, dst: BasicIRBlock, block: BasicIRBlock):
                seen = set([block])
                remaining = [src]
                while remaining:
                    node, remaining = remaining[0], remaining[1:]
                    if node == dst:
                        return False
                    if node in seen:
                        continue
                    seen.add(node)
                    remaining.extend(node.outgoing)
                return True

            options = []
            for pred in dst.incoming:
                if src in pred.dominators:
                    if is_path_blocker(src, dst, pred):
                        options.append(pred)
            assert (
                len(options) == 1
            ), f"got zero or multiple options: src={src.id}, dst={dst.id}, options->[{', '.join(str(option.id) for option in options)}]"
            return options[0]

        # print_basic_blocks(self.basic_blocks)

        # Second pass: remap phi node source information.
        for block in self.basic_blocks:
            for i in range(len(block.ops)):
                op = block.ops[i]
                if not isinstance(op, TransitionalPhiOp):
                    continue

                def map_src(src):
                    reg, src = src
                    if src is None and isinstance(reg, SlotRegister):
                        return (reg, source_block_to_predecessor(self.entry_basic_block, block))
                    if src is None:
                        src = self.find_writer(self.ir, reg)
                    if isinstance(src, IROp):
                        src_block = lookup_tree_ir_op(src)
                        pred_block = source_block_to_predecessor(src_block, block)
                        return (reg, pred_block)
                    raise AssertionError(f"invalid source: {src}")

                block.ops[i] = BasicBlockPhiOp(
                    dst=op.dst, srcs=[map_src(src) for src in op.srcs], span=op.span
                )

    def simplify_control_flow_graph(self) -> bool:
        # Any basic block B with a single incoming edge can be merged into its predecessor,
        # as long as that predecessor either only has a single outgoing edge (in which case the
        # ops can be merged), or B consists of just a jump (in which case this is effectively
        # just renaming B to its own successor).
        # TODO: handle the latter case
        # (Note: dead code elimination will take care of blocks with _no_ incoming edges; this
        # requires some more care, since output registers of the unreachable basic block need
        # to be pruned in later blocks.)

        any_change = False

        # Keep track of which block each about-to-be-deleted block was merged into, so we can
        # fix up any phi node source references.
        combining_map = {}
        for block in self.basic_blocks:
            if len(block.incoming) == 1:
                pred = list(block.incoming)[0]
                if len(pred.outgoing) != 1:
                    continue
                assert list(pred.outgoing)[0] == block

                any_change = True

                # Delete whatever caused the jump at the end of the previous block.
                # (Could be a basic jump op, or for instance a multimethod dispatch that has
                # been optimized / simplified so we know it has only one branch.)
                pred.ops = pred.ops[:-1] + block.ops
                pred.outgoing.remove(block)

                for succ in block.outgoing:
                    assert block in succ.incoming
                    succ.incoming.remove(block)
                    pred.link_outgoing(succ)

                combining_map[block] = pred

        for block in self.basic_blocks:
            for op in block.ops:
                if not isinstance(op, BasicBlockPhiOp):
                    continue

                def walk(b):
                    # May take several steps, if we deleted multiple blocks in a linear chain.
                    while b in combining_map:
                        b = combining_map[b]
                    return b

                op.srcs = [(reg, walk(src_block)) for reg, src_block in op.srcs]

        print(
            colored(
                f"squashed blocks: {', '.join(str(block.id) for block in combining_map.keys())}",
                "cyan",
            )
        )
        for block in combining_map.keys():
            self.basic_blocks.remove(block)

        if any_change:
            self.compute_dominators()

        return any_change

    def compute_dominators(self) -> None:
        reachable_blocks = self.find_reachable_blocks()

        for block in self.basic_blocks:
            if block is self.entry_basic_block:
                block.dominators = set([block])
            else:
                block.dominators = set(self.basic_blocks)

        any_change = True
        while any_change:
            any_change = False
            for block in self.basic_blocks:
                if block is self.entry_basic_block:
                    continue
                if block in reachable_blocks:
                    reachable_preds = block.incoming & reachable_blocks
                else:
                    reachable_preds = block.incoming
                if reachable_preds:
                    new_dominators = set([block]) | set.intersection(
                        *[pred.dominators for pred in reachable_preds]
                    )
                else:
                    new_dominators = set([block])
                if new_dominators != block.dominators:
                    any_change = True
                    block.dominators = new_dominators

    def find_reachable_blocks(self) -> set[BasicIRBlock]:
        # A block is unreachable if there is no path from the entry block to it.
        reachable_blocks = set()
        queue = [self.entry_basic_block]
        while queue:
            block, queue = queue[0], queue[1:]
            reachable_blocks.add(block)
            for succ in block.outgoing:
                if succ not in reachable_blocks:
                    queue.append(succ)
        return reachable_blocks

    def cfg_eliminate_dead_code(self) -> bool:
        # This function:
        # * Deletes unreachable blocks (and makes sure to delete from the rest of the reachable
        #   blocks any use of registers which were generated from unreachable blocks).
        # * Deletes unreachable _ops_ in reachable blocks (ops which come after tail calls).
        # * Deletes any ops whose output is not a source register of any reachable op.
        #   (Caveat: for ops with side effects, just delete the destination register instead of
        #   the whole op.)

        reachable_blocks = self.find_reachable_blocks()
        unreachable_blocks = set(self.basic_blocks) - reachable_blocks
        # Sanity check: no reachable block should have an outgoing edge to an unreachable block.
        # (Unreachable blocks may have outgoing edges to reachable blocks, though.)
        for block in reachable_blocks:
            assert not (block.outgoing & unreachable_blocks)

        # Remove all the outgoing links to reachable blocks, and delete from the CFG
        for block in unreachable_blocks:
            for succ in block.outgoing:
                succ.incoming.remove(block)
        for block in unreachable_blocks:
            self.basic_blocks.remove(block)

        # List of registers used by reachable code.
        used_regs = set()
        # List of registers which are outputs of unreachable code.
        unassigned_regs = set()

        # We already know that if there are unreachable blocks, we're going to delete those.
        any_change = len(unreachable_blocks) > 0

        for block in reachable_blocks:
            beyond_tail_call = False
            for op in block.ops:
                if beyond_tail_call:
                    # These ops will be deleted.
                    any_change = True
                    if op.dst:
                        unassigned_regs.add(op.dst)
                else:
                    used_regs |= basic_ir_op_reg_srcs(op)
                    if isinstance(op, BaseInvokeOp) and op.tail_call:
                        beyond_tail_call = True
                        if op.dst:
                            unassigned_regs.add(op.dst)

        for block in unreachable_blocks:
            for op in block.ops:
                if op.dst:
                    unassigned_regs.add(op.dst)

        # Now delete unreachable ops from reachable blocks.
        for block in reachable_blocks:
            for i in range(len(block.ops)):
                op = block.ops[i]
                if isinstance(op, BaseInvokeOp) and op.tail_call:
                    block.ops = block.ops[: i + 1]
                    # Unlink this block from successors.
                    for succ in block.outgoing:
                        succ.incoming.remove(block)
                    block.outgoing.clear()
                    break

        # That could have made other blocks unreachable.
        reachable_blocks = self.find_reachable_blocks()
        unreachable_blocks = set(self.basic_blocks) - reachable_blocks

        print(
            colored(
                f"unreachable blocks: {', '.join(str(block.id) for block in unreachable_blocks)}",
                "cyan",
            )
        )
        print(colored(f"unassigned regs: {', '.join(str(reg) for reg in unassigned_regs)}", "cyan"))

        for block in reachable_blocks:
            old_ops = block.ops
            block.ops = []
            for op in old_ops:
                # LINEAR OPS
                if isinstance(op, UnreachableOp):
                    raise AssertionError("unreachable op shouldn't exist in reachable code")
                elif isinstance(op, CopyOp):
                    assert op.src not in unassigned_regs
                    if op.dst in used_regs:
                        block.ops.append(op)
                elif isinstance(op, BasicBlockPhiOp):
                    assert not (set(reg for reg, _ in op.srcs) <= unassigned_regs)
                    if op.dst in used_regs:
                        op.srcs = [
                            (reg, src_block)
                            for reg, src_block in op.srcs
                            if reg not in unassigned_regs
                        ]
                        if len(op.srcs) == 1:
                            # Convert to a direct assignment.
                            reg, _ = op.srcs[0]
                            block.ops.append(CopyOp(dst=op.dst, src=reg, span=op.span))
                        else:
                            block.ops.append(op)
                elif isinstance(op, LiteralOp):
                    if op.dst in used_regs:
                        block.ops.append(op)
                elif isinstance(op, BaseInvokeOp):
                    assert not (set(op.call_args) & unassigned_regs)
                    if op.dst not in used_regs:
                        # Can't just delete the op, since this op has side effects.
                        op.dst = None
                    block.ops.append(op)
                elif isinstance(op, ClosureOp):
                    if op.dst in used_regs:
                        block.ops.append(op)
                elif isinstance(op, SlotLookupOp):
                    if op.dst not in used_regs:
                        # Can't just delete the op, since this op has side effects.
                        op.dst = None
                    block.ops.append(op)
                elif isinstance(op, VectorOp):
                    assert not (set(op.components) & unassigned_regs)
                    if op.dst in used_regs:
                        block.ops.append(op)
                elif isinstance(op, TupleOp):
                    assert not (set(op.components) & unassigned_regs)
                    if op.dst in used_regs:
                        block.ops.append(op)
                elif isinstance(op, SignalOp):
                    assert not (set(op.signal_args) & unassigned_regs)
                    if op.dst not in used_regs:
                        # Can't just delete the op, since this op has side effects.
                        op.dst = None
                    block.ops.append(op)

                # NONLINEAR OPS
                elif isinstance(op, BasicBlockJumpOp):
                    block.ops.append(op)
                elif isinstance(op, ReturnOp):
                    assert op.value not in unassigned_regs
                    block.ops.append(op)
                elif isinstance(op, BasicBlockMultimethodDispatchOp):
                    assert not (
                        set([arg for arg in op.dispatch_args if arg is not None]) & unassigned_regs
                    )
                    block.ops.append(op)
                elif isinstance(op, BasicBlockIfElseOp):
                    assert op.condition not in unassigned_regs
                    block.ops.append(op)

                else:
                    raise AssertionError(f"Forgot an op: {type(op)}")

        self.compute_dominators()
        return any_change

    # Inputs:
    #   join:
    #       Combines multiple states (generally, one output state from each predecessor of a block)
    #       into a single state (generally, the new input to that block).
    #       Must be a pure function.
    #   transfer:
    #       Passes a state through a basic block, and outputs a single output value along with a map of
    #       output states, one per successor.
    #       Must be a pure function.
    #   initial_in_state:
    #       Initial block input state, per block.
    #   initial_out_states:
    #       Initial block output states (one per block's successor), per block.
    # Runs a forward dataflow analysis until convergence, then returns the final transfer output of
    #   each block.
    # Note that this is only guaranteed to ever visit blocks reachable from the entry block.
    def forward_dataflow_analysis(
        self,
        join: Callable[[list[S]], S],
        transfer: Callable[[BasicIRBlock, S], Tuple[T, dict[BasicIRBlock, S]]],
        initial_in_state: dict[BasicIRBlock, S],
        initial_out_states: dict[BasicIRBlock, dict[BasicIRBlock, S]],
    ) -> dict[BasicIRBlock, T]:
        # Last known (joined) input state per block.
        last_in_state: dict[BasicIRBlock, S] = initial_in_state
        # Last known output state collection per block: a possibly distinct output state per successor.
        last_out_states: dict[BasicIRBlock, dict[BasicIRBlock, S]] = initial_out_states
        last_transfer_value: dict[BasicIRBlock, T] = {}
        work_queue: list[BasicIRBlock]

        def do_work(block: BasicIRBlock, force_transfer: bool):
            if block is self.entry_basic_block:
                # Don't join the in-state; just use the provided initial values, always.
                in_state = last_in_state[block]
            else:
                in_state = join([last_out_states[pred][block] for pred in block.incoming])
            if last_in_state[block] == in_state and not force_transfer:
                # Nothing changed; no work to do!
                return
            last_in_state[block] = in_state
            value, out_states = transfer(block, in_state)
            last_transfer_value[block] = value
            _last_out_states = last_out_states[block]  # for the current block only
            for succ in block.outgoing:
                if out_states[succ] != _last_out_states[succ]:
                    # Successor's joined input may change; add the successor to the work queue.
                    if succ not in work_queue:
                        work_queue.append(succ)
            last_out_states[block] = out_states

        # Prime the pump:
        work_queue = []
        initial_visited = set()
        initial_visit_queue = [self.entry_basic_block]
        while initial_visit_queue:
            block, initial_visit_queue = initial_visit_queue[0], initial_visit_queue[1:]
            initial_visited.add(block)
            do_work(block, force_transfer=True)
            for succ in block.outgoing:
                if succ not in initial_visited:
                    initial_visit_queue.append(succ)

        # Run the pump:
        while work_queue:
            block, work_queue = work_queue[0], work_queue[1:]
            do_work(block, force_transfer=False)

        return last_transfer_value

    def cfg_infer_types(self) -> None:
        # Per register, indicates knowledge of what possible types the value is known to have,
        # or ANY_TYPE if it could take any type. If a register is not present, we don't yet know
        # what types it may have, and must not use it for typing judgements yet.
        # Note that this implies a _subtyping_ relationship: the value is a subtype of any of these
        # provided types.
        S: TypeAlias = dict[Register, Types]

        def merge_types(many_types: list[Types]) -> Types:
            if any(types is ANY_TYPE for types in many_types):
                return ANY_TYPE
            merged = []
            for types in many_types:
                for type in types:
                    if type not in merged:
                        merged.append(type)
            # TODO: can simplify by deleting any A where A <= B for some other B in the list.
            return merged

        def join(states: list[S]) -> S:
            merged = {}
            for state in states:
                for reg, types in state.items():
                    if reg not in merged:
                        merged[reg] = types
                        continue
                    merged[reg] = merge_types([merged[reg], types])
            return merged

        def transfer(block: BasicIRBlock, state: S) -> dict[BasicIRBlock, S]:
            reg_types = dict(state)
            for op in block.ops:
                # LINEAR OPS
                if isinstance(op, UnreachableOp):
                    raise AssertionError("shouldn't have gotten to unreachable op")
                elif isinstance(op, CopyOp):
                    if op.src in reg_types:
                        reg_types[op.dst] = reg_types[op.src]
                elif isinstance(op, BasicBlockPhiOp):
                    if all(reg in reg_types for reg, _ in op.srcs):
                        reg_types[op.dst] = merge_types(
                            [reg_types.get(reg, []) for reg, _ in op.srcs]
                        )
                elif isinstance(op, LiteralOp):
                    reg_types[op.dst] = [type_of(op.value)]
                elif isinstance(op, BaseInvokeOp):
                    # TODO: use any more return type information?
                    if isinstance(op, InvokeNativeOp) and op.return_type:
                        reg_types[op.dst] = [op.return_type]
                    else:
                        reg_types[op.dst] = ANY_TYPE
                elif isinstance(op, ClosureOp):
                    reg_types[op.dst] = [QuoteType]
                elif isinstance(op, SlotLookupOp):
                    # TODO: put a lease on the slot?
                    reg_types[op.dst] = ANY_TYPE
                elif isinstance(op, VectorOp):
                    # TODO: might want to get fancier in the future with per-element typing.
                    reg_types[op.dst] = [VectorType]
                elif isinstance(op, TupleOp):
                    # TODO: might want to get fancier in the future with per-element typing.
                    reg_types[op.dst] = [TupleType]
                elif isinstance(op, SignalOp):
                    # Signaling really can produce any result -- it's up to the user / debugger.
                    reg_types[op.dst] = ANY_TYPE

                # NONLINEAR OPS -- directly return.
                elif isinstance(op, BasicBlockJumpOp):
                    return reg_types, {op.target: reg_types}
                elif isinstance(op, ReturnOp):
                    return reg_types, {}
                elif isinstance(op, BasicBlockMultimethodDispatchOp):

                    def derive_types(
                        args: list[Optional[Register]], matchers: list[ParameterMatcher]
                    ) -> S:
                        # TODO: could get fancier and even _remove_ types if we know that they are impossible
                        # under the assumption that this dispatch branch is chosen.
                        # For instance, if there are types A < B, and a method 'a' on A and a method 'b' on B,
                        # then if 'b' is chosen it means not only that the argument is a subtype of B, but it
                        # is also _not_ a subtype of A.
                        assert len(args) == len(matchers)
                        restricted_reg_types = dict(reg_types)
                        for arg, matcher in zip(args, matchers):
                            if arg is None or arg not in reg_types:
                                continue
                            if isinstance(matcher, ParameterAnyMatcher):
                                # Can't derive any new info from this.
                                continue
                            elif isinstance(matcher, ParameterTypeMatcher):
                                arg_types = reg_types[arg]
                                arg_supertype = matcher.param_type
                                # Argument type is a subtype of arg_super_type; we can exclude any known possible
                                # types that don't meet this requirement.
                                if arg_types is ANY_TYPE:
                                    restricted = [arg_supertype]
                                else:
                                    # TODO: actually this may be a bit tricky... suppose we have a struct type S
                                    # and an (a priori) unrelated mixin M, i.e. we do not have S <= M. If we know
                                    # that some argument is of type S (or subtype thereof), and the multimethod
                                    # parameter matcher expects an M, this does _not_ let us conclude that there
                                    # are no possible options for this variable type (assuming this dispatch option
                                    # was selected); rather, that conclusion depends on whether or not there is any
                                    # _other_ type which is both a subtype of S and M, such as a new user defined
                                    # type which simply inherits from both.
                                    # Maybe the right thing for now is to consider types as a conjuction of
                                    # disjunctions -- in this case we would 'simply' note that the variable must be
                                    # both an S and an M, with no further simplification.
                                    # TODO: apply intersection of arg_supertype and existing arg_types.
                                    restricted = [arg_supertype]
                                restricted_reg_types[arg] = restricted
                            else:
                                continue
                        return restricted_reg_types

                    reg_types_per_succ = {
                        method_block: derive_types(op.dispatch_args, method.param_matchers)
                        for method, method_block in op.dispatch
                    }
                    # Likewise could get fancier here with restricted type options. However, probably
                    # not very useful, since these branch just lead to a signaling ops and are not
                    # the happy path.
                    if op.no_matching_method:
                        reg_types_per_succ[op.no_matching_method] = reg_types
                    if op.ambiguous_method_resolution:
                        reg_types_per_succ[op.ambiguous_method_resolution] = reg_types
                    return reg_types, reg_types_per_succ
                elif isinstance(op, BasicBlockIfElseOp):
                    # TODO: depending on what the condition is, we may be able to apply path dependent typing.
                    # For now, just apply the same type knowledge to both branches.
                    return reg_types, {op.true_block: reg_types, op.false_block: reg_types}
                else:
                    raise AssertionError(f"Forgot an op: {type(op)}")

            # If we got this fartransfer(block, in_state), there was no nonlinear op at the end. Must be a top-level block being compiled.
            assert not block.outgoing
            return reg_types, {}

        initial_in_state: dict[BasicIRBlock, S] = {block: {} for block in self.basic_blocks}
        initial_out_states: dict[BasicIRBlock, dict[BasicIRBlock, S]] = {
            block: {succ: {} for succ in block.outgoing} for block in self.basic_blocks
        }

        for block in self.basic_blocks:
            if block is not self.entry_basic_block:
                continue
            reg_types = initial_in_state[block]
            for reg, param_type in self.parameter_types.items():
                if param_type:
                    reg_types[reg] = [param_type]
                else:
                    reg_types[reg] = ANY_TYPE

        typing_info = self.forward_dataflow_analysis(
            join, transfer, initial_in_state, initial_out_states
        )
        for block in self.basic_blocks:
            if block in typing_info:
                block.types = typing_info[block]

            # print(f"===== block {block.id} typing inputs ======")
            # if block not in typing_info:
            #     print("(unreachable)")
            #     continue
            # reg_types = typing_info[block]
            # for reg, types in reg_types.items():
            #     if types is None:
            #         print(f"{reg} -> unreachable / never evaluated")
            #     elif types is ANY_TYPE:
            #         continue
            #     else:
            #         print(f"{reg} -> " + ", ".join(str(t) for t in types))

    def cfg_simplify_dispatches(self) -> bool:
        any_change = False

        for block in self.basic_blocks:
            op = block.ops[-1]
            if not isinstance(op, BasicBlockMultimethodDispatchOp):
                continue

            def is_certain_match(reg: Register, matcher: ParameterMatcher):
                if isinstance(matcher, ParameterAnyMatcher):
                    return True
                elif isinstance(matcher, ParameterTypeMatcher):
                    reg_types = block.types[reg]
                    if reg_types is ANY_TYPE:
                        return False
                    if not all(is_subtype(reg_type, matcher.param_type) for reg_type in reg_types):
                        return False
                    return True
                elif isinstance(matcher, ParameterValueMatcher):
                    # TODO: would be nice to evaluate these as well, if we have value-level 'type'
                    # information for registers.
                    return False
                else:
                    raise AssertionError(f"forgot a matcher type: {matcher}")

            def is_certain_mismatch(reg: Register, matcher: ParameterMatcher):
                # TODO: implement.
                return False

            def method_is_certain_match(
                reg_and_matchers: list[Tuple[Register, ParameterMatcher]]
            ) -> bool:
                return all(is_certain_match(reg, matcher) for reg, matcher in reg_and_matchers)

            def method_is_certain_mismatch(
                reg_and_matchers: list[Tuple[Register, ParameterMatcher]]
            ) -> bool:
                return all(is_certain_mismatch(reg, matcher) for reg, matcher in reg_and_matchers)

            dispatches_to_delete = []
            for i, method_and_block in enumerate(op.dispatch):
                method, _ = method_and_block
                assert len(method.param_matchers) == len(op.dispatch_args)
                reg_and_matchers = [
                    (reg, matcher)
                    for reg, matcher in zip(op.dispatch_args, method.param_matchers)
                    if reg is not None
                ]
                if method_is_certain_match(reg_and_matchers):
                    # We can delete no-matching-method error handling, yay!
                    if op.no_matching_method:
                        any_change = True
                        block.unlink_outgoing(op.no_matching_method)
                        op.no_matching_method = None
                if method_is_certain_mismatch(reg_and_matchers):
                    dispatches_to_delete.append(method_and_block)

            if dispatches_to_delete:
                any_change = True
                for option in dispatches_to_delete:
                    op.dispatch.remove(option)

            # If dispatch actually doesn't care what the value of a particular parameter is, then
            # delete that parameter.
            for i, param_reg in enumerate(op.dispatch_args):
                if not param_reg:
                    continue
                reg_required = False
                for method, _ in op.dispatch:
                    if not is_certain_match(param_reg, method.param_matchers[i]):
                        reg_required = True
                        break
                if not reg_required:
                    any_change = True
                    op.dispatch_args[i] = None

            # TODO: can get fancier with the precondition here, if we know types well enough.
            if len(op.dispatch) == 1:
                # We can delete ambiguous-method-resolution handling, yay!
                if op.ambiguous_method_resolution:
                    any_change = True
                    block.unlink_outgoing(op.ambiguous_method_resolution)
                    op.ambiguous_method_resolution = None

            # Sanity check -- we shouldn't have been able to optimize away every option.
            assert op.dispatch or op.no_matching_method or op.ambiguous_method_resolution

            # The Final Optimization To End All Optimizations:
            # If the dispatch actually has only one option, just make it a jump instead.
            if (
                len(op.dispatch) == 1
                and not op.no_matching_method
                and not op.ambiguous_method_resolution
            ):
                _, target = op.dispatch[0]
                block.ops[-1] = BasicBlockJumpOp(target=target, span=op.span)
                any_change = True
            elif not op.dispatch and op.no_matching_method and not op.ambiguous_method_resolution:
                block.ops[-1] = BasicBlockJumpOp(target=op.no_matching_method, span=op.span)
                any_change = True
            elif not op.dispatch and not op.no_matching_method and op.ambiguous_method_resolution:
                block.ops[-1] = BasicBlockJumpOp(
                    target=op.ambiguous_method_resolution, span=op.span
                )
                any_change = True

        if any_change:
            self.compute_dominators()
        return any_change

    def cfg_propagate_copies(self) -> bool:
        # Remapping is from destination to source in various assignments.
        remapping: dict[Register, Register] = {}
        for block in self.basic_blocks:
            for op in block.ops:
                if not isinstance(op, CopyOp):
                    continue
                remapping[op.dst] = remapping.get(op.src, op.src)

        any_change = False

        def remap(reg: Register) -> Register:
            nonlocal any_change
            remapped = remapping.get(reg, reg)
            if remapped != reg:
                any_change = True
            return remapped

        for block in self.basic_blocks:
            for op in block.ops:
                if isinstance(op, UnreachableOp):
                    pass
                elif isinstance(op, CopyOp):
                    op.src = remap(op.src)
                elif isinstance(op, BasicBlockPhiOp):
                    op.srcs = [(remap(reg), block) for reg, block in op.srcs]
                elif isinstance(op, LiteralOp):
                    pass
                elif isinstance(op, InvokeRegisterOp):
                    op.callable = remap(op.callable)
                    op.call_args = [remap(arg) for arg in op.call_args]
                elif isinstance(op, InvokeMultimethodOp):
                    op.call_args = [remap(arg) for arg in op.call_args]
                elif isinstance(op, InvokeMethodOp):
                    op.call_args = [remap(arg) for arg in op.call_args]
                elif isinstance(op, InvokeQuoteOp):
                    op.call_args = [remap(arg) for arg in op.call_args]
                elif isinstance(op, InvokeIntrinsicOp):
                    op.call_args = [remap(arg) for arg in op.call_args]
                elif isinstance(op, InvokeNativeOp):
                    op.call_args = [remap(arg) for arg in op.call_args]
                elif isinstance(op, ClosureOp):
                    pass
                elif isinstance(op, SlotLookupOp):
                    pass
                elif isinstance(op, VectorOp):
                    op.components = [remap(comp) for comp in op.components]
                elif isinstance(op, TupleOp):
                    op.components = [remap(comp) for comp in op.components]
                elif isinstance(op, SignalOp):
                    op.signal_args = [remap(arg) for arg in op.signal_args]
                elif isinstance(op, BasicBlockJumpOp):
                    pass
                elif isinstance(op, ReturnOp):
                    op.value = remap(op.value)
                elif isinstance(op, BasicBlockMultimethodDispatchOp):
                    op.dispatch_args = [remap(arg) for arg in op.dispatch_args]
                elif isinstance(op, BasicBlockIfElseOp):
                    op.condition = remap(op.condition)
                else:
                    raise AssertionError(f"forgot an IROp: {type(op)}")

        return any_change

    def compile_to_low_level_blocks(self) -> None:
        # Maps each basic block to its corresponding generated LR block.
        basic_block_to_lr_block = {}
        # Maps fake positions (negative numbers, which are converted to indices in this list
        # as -fake_target - 1) to basic blocks, so we can fix up jump targets in a second pass
        # once we have generated LR blocks.
        lr_op_fixups = []

        def gen_jump(block: BasicIRBlock) -> int:
            lr_op_fixups.append(block)
            return -len(lr_op_fixups)

        def copy_to_phis(block: BasicIRBlock, successor: BasicIRBlock, span: SourceSpan):
            assert successor in block.outgoing
            for op in successor.ops:
                if not isinstance(op, BasicBlockPhiOp):
                    continue
                for reg, src_block in op.srcs:
                    if src_block == block:
                        add_lr_op(CopyLROp(dst=op.dst, src=reg, span=span))

        assert not self.lr_blocks
        assert not self.entry_lr_block

        for block in self.basic_blocks:
            lr_block = LRBlock(id=len(self.lr_blocks), ops=[], incoming=set(), outgoing=set())
            self.lr_blocks.append(lr_block)
            basic_block_to_lr_block[block] = lr_block

            def add_lr_op(op: LROp):
                assert isinstance(op, LROp)
                lr_block.ops.append(op)
                return op

            for op in block.ops:
                if isinstance(op, UnreachableOp):
                    raise AssertionError("shouldn't get unreachable op here")
                elif isinstance(op, CopyOp):
                    add_lr_op(CopyLROp(dst=op.dst, src=op.src, span=op.span))
                elif isinstance(op, BasicBlockPhiOp):
                    # Actually, do nothing here. Must be handled already by copy_to_phis() in
                    # our predecessor blocks.
                    pass
                elif isinstance(op, LiteralOp):
                    add_lr_op(LiteralLROp(dst=op.dst, value=op.value, span=op.span))
                elif isinstance(
                    op, (InvokeRegisterOp, InvokeMultimethodOp, InvokeIntrinsicOp, InvokeNativeOp)
                ):
                    for arg in op.call_args:
                        add_lr_op(PushLROp(src=arg, span=op.span))
                    if isinstance(op, InvokeRegisterOp):
                        add_lr_op(
                            InvokeRegisterLROp(
                                callable=op.callable,
                                num_args=len(op.call_args),
                                tail_call=op.tail_call,
                                span=op.span,
                            )
                        )
                    elif isinstance(op, InvokeMultimethodOp):
                        add_lr_op(
                            InvokeMultimethodLROp(
                                multimethod=op.multimethod,
                                num_args=len(op.call_args),
                                tail_call=op.tail_call,
                                span=op.span,
                            )
                        )
                    elif isinstance(op, InvokeIntrinsicOp):
                        add_lr_op(
                            InvokeIntrinsicLROp(
                                intrinsic=op.intrinsic,
                                num_args=len(op.call_args),
                                tail_call=op.tail_call,
                                span=op.span,
                            )
                        )
                    elif isinstance(op, InvokeNativeOp):
                        add_lr_op(
                            InvokeNativeLROp(
                                native=op.native,
                                num_args=len(op.call_args),
                                tail_call=op.tail_call,
                                span=op.span,
                            )
                        )
                    else:
                        raise AssertionError("shouldn't get here")
                    if not op.tail_call:
                        if op.dst:
                            add_lr_op(PopLROp(dst=op.dst, span=op.span))
                        else:
                            add_lr_op(DropLROp(span=op.span))
                elif isinstance(op, ClosureOp):
                    add_lr_op(ClosureLROp(dst=op.dst, quote=op.quote, span=op.span))
                elif isinstance(op, SlotLookupOp):
                    add_lr_op(SlotLookupLROp(dst=op.dst, slot_name=op.slot_name, span=op.span))
                elif isinstance(op, VectorOp):
                    add_lr_op(VectorLROp(dst=op.dst, components=op.components, span=op.span))
                elif isinstance(op, TupleOp):
                    add_lr_op(VectorLROp(dst=op.dst, components=op.components, span=op.span))
                elif isinstance(op, SignalOp):
                    add_lr_op(
                        SignalLROp(
                            dst=op.dst,
                            condition_name=op.condition_name,
                            signal_args=op.signal_args,
                            span=op.span,
                        )
                    )
                elif isinstance(op, BasicBlockJumpOp):
                    # TODO: ideally in this case we don't need to add copy ops, since we could have
                    # just remapped the originating ops' destinations according to what we would
                    # otherwise copy here.
                    copy_to_phis(block, op.target, span=op.span)
                    add_lr_op(JumpLROp(target=gen_jump(op.target), span=op.span))
                elif isinstance(op, ReturnOp):
                    add_lr_op(ReturnLROp(src=op.value, span=op.span))
                elif isinstance(op, BasicBlockMultimethodDispatchOp):
                    keep_indices = [
                        i for i in range(len(op.dispatch_args)) if op.dispatch_args[i] is not None
                    ]

                    dispatch_args = [op.dispatch_args[i] for i in keep_indices]
                    dispatch = []
                    dispatch_op: MultimethodDispatchLROp = add_lr_op(
                        MultimethodDispatchLROp(
                            slot_name=op.slot_name,
                            dispatch_args=dispatch_args,
                            dispatch=dispatch,
                            no_matching_method=None,
                            ambiguous_method_resolution=None,
                            span=op.span,
                        )
                    )
                    for method, method_block in op.dispatch:
                        dispatch_op.dispatch.append(
                            (
                                [method.param_matchers[i] for i in keep_indices],
                                len(lr_block.ops),
                            )
                        )
                        copy_to_phis(block, method_block, span=op.span)
                        add_lr_op(JumpLROp(target=gen_jump(method_block), span=op.span))

                    if op.no_matching_method:
                        dispatch_op.no_matching_method = len(lr_block.ops)
                        copy_to_phis(block, op.no_matching_method, span=op.span)
                        add_lr_op(JumpLROp(target=gen_jump(op.no_matching_method), span=op.span))

                    if op.ambiguous_method_resolution:
                        dispatch_op.ambiguous_method_resolution = len(lr_block.ops)
                        copy_to_phis(block, op.ambiguous_method_resolution, span=op.span)
                        add_lr_op(
                            JumpLROp(target=gen_jump(op.ambiguous_method_resolution), span=op.span)
                        )
                elif isinstance(op, BasicBlockIfElseOp):
                    # if not <condition> goto F
                    # copy into phi nodes for <true-block>
                    # goto <true-block>
                    # F:
                    # copy into phi nodes for <false-block>
                    # goto <false-block>
                    conditional_jump = add_lr_op(
                        ConditionalJumpLROp(
                            condition=op.condition, match=False, target=-1, span=op.span
                        )
                    )
                    copy_to_phis(block, op.true_block, span=op.span)
                    add_lr_op(JumpLROp(target=gen_jump(op.true_block), span=op.span))
                    conditional_jump.target = len(lr_block.ops)
                    copy_to_phis(block, op.false_block, span=op.span)
                    add_lr_op(JumpLROp(target=gen_jump(op.false_block), span=op.span))
                else:
                    raise AssertionError(f"forgot an IROp: {type(op)}")

        # Fix up jump targets.
        for block in self.lr_blocks:
            for op in block.ops:
                if not isinstance(op, JumpLROp):
                    continue
                if op.target < 0:
                    op.target = basic_block_to_lr_block[lr_op_fixups[-op.target - 1]]

        # Add other graph information to the LR blocks.
        self.entry_lr_block = basic_block_to_lr_block[self.entry_basic_block]
        for block in self.basic_blocks:
            lr_block = basic_block_to_lr_block[block]
            lr_block.incoming = set(basic_block_to_lr_block[pred] for pred in block.incoming)
            lr_block.outgoing = set(basic_block_to_lr_block[succ] for succ in block.outgoing)


should_show_compiler_output = False
indent_per_level = "    "


def print_op(index: int, op: IROp, depth: int):
    level = indent_per_level
    indent = level * depth

    def prefix():
        # print(indent + f"{index}: ", end="")
        print(indent, end="")

    if isinstance(op, InlineBlockOp):
        prefix()
        print("<inline block>:")
        (inline_block,) = op.sub_blocks
        print(indent + level + colored("slots:", "grey"))
        for name, value in inline_block.ctxt.slots:
            print(indent + level + level + colored(f"{name} -> {value}", "grey"))
        print_ir_block(inline_block, depth=(depth + 1))
    elif isinstance(op, Label):
        prefix()
        print(colored(f"{op.id}:", "light_blue"))
    elif isinstance(op, JumpOp):
        prefix()
        print("jump " + colored(f"{op.target.id}", "light_magenta"))
    elif isinstance(op, UnreachableOp):
        prefix()
        if op.dst:
            print(f"{op.dst} = unreachable")
        else:
            print("unreachable")
    elif isinstance(op, ReturnOp):
        prefix()
        print(f"return {op.value}")
    elif isinstance(op, CopyOp):
        prefix()
        print(f"{op.dst} = {op.src}")
    elif isinstance(op, PhiOp):
        prefix()
        print(f"{op.dst} = phi({', '.join(str(reg) for reg, _ in op.srcs)})", end="")
        if op.force_keep:
            print(" (force-keep)")
        else:
            print()
    elif isinstance(op, LiteralOp):
        prefix()
        print(
            f"{op.dst} = literal {repr(op.value.value) if isinstance(op.value, StringValue) else op.value}"
        )
    elif isinstance(op, BaseInvokeOp):
        prefix()

        if op.dst:
            print(f"{op.dst} = ", end="")
        if op.tail_call:
            print("tail-", end="")

        if isinstance(op, InvokeRegisterOp):
            print(f"invoke {op.callable}", end="")
        elif isinstance(op, InvokeMultimethodOp):
            print(
                f"invoke-multimethod {op.multimethod.name} (inline_dispatch={op.multimethod.inline_dispatch})",
                end="",
            )
        elif isinstance(op, InvokeMethodOp):
            print("invoke-method ", end="")
            if isinstance(op.method.body, QuoteMethodBody):
                print(f"<quote:{op.method.body}>", end="")
            elif isinstance(op.method.body, IntrinsicMethodBody):
                print(f"<intrinsic:{op.method.body.handler.handler}>", end="")
            elif isinstance(op.method.body, NativeMethodBody):
                print(f"<native:{op.method.body.handler.handler}>", end="")
            else:
                raise AssertionError(f"forgot a method body type: {op.method.body}")
            print(f" (label={op.method_start_label.id}, inline={op.method.inline})", end="")
        elif isinstance(op, InvokeQuoteOp):
            print(f"invoke-quote {op.quote}", end="")
        elif isinstance(op, InvokeIntrinsicOp):
            print(f"invoke-intrinsic {op.intrinsic.handler}", end="")
        elif isinstance(op, InvokeNativeOp):
            print(f"invoke-native {op.native.handler}", end="")
            if op.return_type:
                print(f" (returns {op.return_type})", end="")
        else:
            raise AssertionError(f"forgot an op type: {op}")

        if op.call_args:
            print(f" with {', '.join(str(arg) for arg in op.call_args)}", end="")
        else:
            print(" with <no args>", end="")
        if op.tail_position:
            print(" (in tail position)")
        else:
            print()
    elif isinstance(op, ClosureOp):
        prefix()
        print(f"{op.dst} = closure {op.quote}")
    elif isinstance(op, SlotLookupOp):
        prefix()
        print(f"{op.dst} = slot-lookup {op.slot_name}")
    elif isinstance(op, VectorOp):
        prefix()
        print(f"{op.dst} = vector {', '.join(str(comp) for comp in op.components)}")
    elif isinstance(op, TupleOp):
        prefix()
        print(f"{op.dst} = tuple {', '.join(str(comp) for comp in op.components)}")
    elif isinstance(op, MultimethodDispatchOp):
        prefix()
        print(
            f"multimethod-dispatch {op.slot_name} on {', '.join(str(arg) for arg in op.dispatch_args)}"
        )
        for method, body_block in op.dispatch:

            def matcher_str(matcher):
                if isinstance(matcher, ParameterAnyMatcher):
                    return "<any>"
                elif isinstance(matcher, ParameterTypeMatcher):
                    return matcher.param_type.name
                elif isinstance(matcher, ParameterValueMatcher):
                    return f"eq({matcher.param_value})"

            print(
                indent
                + level
                + f"if ({', '.join(matcher_str(matcher) for matcher in method.param_matchers)}):"
            )
            print_ir_block(body_block, depth=(depth + 2))

        if op.no_matching_method:
            print(indent + level + "no-matching-method:")
            print_ir_block(op.no_matching_method, depth=(depth + 2))
        if op.ambiguous_method_resolution:
            print(indent + level + "ambiguous-method-resolution:")
            print_ir_block(op.ambiguous_method_resolution, depth=(depth + 2))
    elif isinstance(op, SignalOp):
        prefix()
        print(
            f"{op.dst} = signal :{op.condition_name} with {', '.join(str(arg) for arg in op.signal_args)}"
        )
    elif isinstance(op, IfElseOp):
        assert not op.dst
        prefix()
        print(f"if-else {op.condition}")
        print(indent + level + "if-truthy:")
        print_ir_block(op.true_block, depth=(depth + 2))
        print(indent + level + "if-falsy:")
        print_ir_block(op.false_block, depth=(depth + 2))
    elif isinstance(op, BasicBlockJumpOp):
        prefix()
        print(f"jump {op.target.id}")
    elif isinstance(op, TransitionalPhiOp):
        prefix()
        print(f"{op.dst} = phi({', '.join(str(reg) for reg, _ in op.srcs)})")
    elif isinstance(op, BasicBlockPhiOp):
        prefix()

        def src_to_str(src):
            reg, block = src
            if block:
                return f"{block.id}->{reg}"
            else:
                return f"{reg}"

        print(f"{op.dst} = phi({', '.join(src_to_str(src) for src in op.srcs)})")
    elif isinstance(op, BasicBlockMultimethodDispatchOp):
        prefix()
        print(
            f"multimethod-dispatch {op.slot_name} on {', '.join(str(arg) if arg else '*' for arg in op.dispatch_args)}"
        )
        for method, target in op.dispatch:

            def matcher_str(matcher):
                if isinstance(matcher, ParameterAnyMatcher):
                    return "<any>"
                elif isinstance(matcher, ParameterTypeMatcher):
                    return matcher.param_type.name
                elif isinstance(matcher, ParameterValueMatcher):
                    return f"eq({matcher.param_value})"

            print(
                indent
                + level
                + f"({', '.join(matcher_str(matcher) for matcher in method.param_matchers)}) -> jump {target.id}"
            )

        if op.no_matching_method:
            print(indent + level + f"no-matching-method -> jump {op.no_matching_method.id}")
        if op.ambiguous_method_resolution:
            print(
                indent
                + level
                + f"ambiguous-method-resolution -> jump {op.ambiguous_method_resolution.id}"
            )
    elif isinstance(op, BasicBlockIfElseOp):
        assert not op.dst
        prefix()
        print(
            f"if-truthy {op.condition} -> jump {op.true_block.id}; else -> jump {op.false_block.id}"
        )
    else:
        raise AssertionError(f"forgot an IROp: {type(op)}")


def print_ir_block(block: TreeIRBlock, depth: int = 0):
    for i, op in enumerate(block.ops):
        print_op(i, op, depth)


def basic_ir_op_reg_srcs(op: IROp) -> set[Register]:
    if isinstance(op, UnreachableOp):
        return set()
    elif isinstance(op, ReturnOp):
        return set([op.value])
    elif isinstance(op, CopyOp):
        return set([op.src])
    elif isinstance(op, LiteralOp):
        return set()
    elif isinstance(op, BaseInvokeOp):
        refs = set(op.call_args)
        if isinstance(op, InvokeRegisterOp):
            refs |= set([op.callable])
        return refs
    elif isinstance(op, ClosureOp):
        return set()
    elif isinstance(op, SlotLookupOp):
        return set()
    elif isinstance(op, VectorOp):
        return set(op.components)
    elif isinstance(op, TupleOp):
        return set(op.components)
    elif isinstance(op, SignalOp):
        return set(op.signal_args)
    elif isinstance(op, BasicBlockJumpOp):
        return set()
    elif isinstance(op, BasicBlockPhiOp):
        return set([reg for reg, _ in op.srcs])
    elif isinstance(op, BasicBlockMultimethodDispatchOp):
        return set([arg for arg in op.dispatch_args if arg is not None])
    elif isinstance(op, BasicBlockIfElseOp):
        return set([op.condition])
    else:
        raise AssertionError(f"forgot an IROp: {type(op)}")


def basic_ir_op_reg_refs(op: IROp) -> set[Register]:
    # dst-set
    return set([op.dst] if op.dst else []) | basic_ir_op_reg_srcs(op)


def print_basic_blocks(blocks: list[BasicIRBlock]):
    for block in sorted(blocks, key=lambda b: b.id):
        print(colored(f"===== block {block.id} =====", "green"))
        print(
            colored(
                "incoming: "
                + ", ".join(str(pred.id) for pred in sorted(block.incoming, key=lambda b: b.id)),
                "grey",
            )
        )
        print(
            colored(
                "dominators: "
                + ", ".join(str(dom.id) for dom in sorted(block.dominators, key=lambda b: b.id)),
                "grey",
            )
        )
        if block.types:
            print(colored("register types:", "grey"))
            referenced_regs = set().union(*[basic_ir_op_reg_refs(op) for op in block.ops])
            for reg in sorted(
                referenced_regs, key=lambda r: ((0 if isinstance(r, SlotRegister) else 1), r.index)
            ):
                if reg not in block.types:
                    print(colored(f"  {reg} -> unreachable / never evaluated", "red"))
                    continue
                types = block.types[reg]
                if types is ANY_TYPE:
                    print(colored(f"  {reg} -> *", "grey"))
                else:
                    print(colored(f"  {reg} -> " + ", ".join(str(t) for t in types), "grey"))
        for i, op in enumerate(block.ops):
            print_op(i, op, depth=0)
        print(
            colored(
                "outgoing: "
                + ", ".join(str(succ.id) for succ in sorted(block.outgoing, key=lambda b: b.id)),
                "grey",
            )
        )


def print_lr_op(op: LROp):
    def target_to_str(target: Union[int, LRBlock]) -> str:
        if isinstance(target, LRBlock):
            return str(target.id)
        else:
            return f"({target})"

    if isinstance(op, CopyLROp):
        print(f"{op.dst} = {op.src}")
    elif isinstance(op, LiteralLROp):
        print(
            f"{op.dst} = literal {repr(op.value.value) if isinstance(op.value, StringValue) else op.value}"
        )
    elif isinstance(op, PushLROp):
        print(f"push {op.src}")
    elif isinstance(op, PopLROp):
        print(f"{op.dst} = pop")
    elif isinstance(op, DropLROp):
        print(f"drop")
    elif isinstance(op, InvokeRegisterLROp):
        if op.tail_call:
            print("tail-", end="")
        print(f"invoke-reg {op.callable} {op.num_args}")
    elif isinstance(op, InvokeMultimethodLROp):
        if op.tail_call:
            print("tail-", end="")
        print(f"invoke-multi {op.multimethod.name} {op.num_args}")
    elif isinstance(op, InvokeIntrinsicLROp):
        if op.tail_call:
            print("tail-", end="")
        print(f"invoke-intrinsic {op.intrinsic.handler} {op.num_args}")
    elif isinstance(op, InvokeNativeLROp):
        if op.tail_call:
            print("tail-", end="")
        print(f"invoke-native {op.native.handler} {op.num_args}")
    elif isinstance(op, ClosureLROp):
        print(f"{op.dst} = closure {op.quote}")
    elif isinstance(op, SlotLookupLROp):
        print(f"{op.dst} = lookup-slot {op.slot_name}")
    elif isinstance(op, VectorLROp):
        print(f"{op.dst} = vector {', '.join(str(comp) for comp in op.components)}")
    elif isinstance(op, TupleLROp):
        print(f"{op.dst} = tuple {', '.join(str(comp) for comp in op.components)}")
    elif isinstance(op, SignalLROp):
        print(f"{op.dst} = signal {op.condition_name}", end="")
        if op.signal_args:
            print(" with " + ", ".join(str(arg) for arg in op.signal_args))
        else:
            print()
    elif isinstance(op, JumpLROp):
        print(f"jump {target_to_str(op.target)}")
    elif isinstance(op, ReturnLROp):
        print(f"return {op.src}")
    elif isinstance(op, ConditionalJumpLROp):
        print(f"jump-{'true' if op.match else 'false'} {op.condition} {target_to_str(op.target)}")
    elif isinstance(op, MultimethodDispatchLROp):

        def matcher_str(matcher):
            if isinstance(matcher, ParameterAnyMatcher):
                return "<any>"
            elif isinstance(matcher, ParameterTypeMatcher):
                return matcher.param_type.name
            elif isinstance(matcher, ParameterValueMatcher):
                return f"eq({matcher.param_value})"

        print(
            f"multimethod-dispatch {op.slot_name} on {', '.join(str(arg) for arg in op.dispatch_args)}"
        )
        for matchers, target in op.dispatch:
            print(
                f"  ({', '.join(matcher_str(matcher) for matcher in matchers)}) -> jump {target_to_str(target)}"
            )
        if op.no_matching_method:
            print(f"  no-matching-method -> jump {target_to_str(op.no_matching_method)}")
        if op.ambiguous_method_resolution:
            print(
                f"  ambiguous-method-resolution -> jump {target_to_str(op.ambiguous_method_resolution)}"
            )
    else:
        raise AssertionError(f"unknown lr-op {op}")


def print_low_level_blocks(blocks: list[LRBlock]):
    print(colored("low level blocks:", "grey"))
    for block in sorted(blocks, key=lambda b: b.id):
        print(colored(f"===== block {block.id} =====", "green"))
        print(
            colored(
                "incoming: "
                + ", ".join(str(pred.id) for pred in sorted(block.incoming, key=lambda b: b.id)),
                "grey",
            )
        )
        for i, op in enumerate(block.ops):
            print(colored(f"({i}) ", "grey"), end="")
            print_lr_op(op)
        print(
            colored(
                "outgoing: "
                + ", ".join(str(succ.id) for succ in sorted(block.outgoing, key=lambda b: b.id)),
                "grey",
            )
        )


def print_low_level_bytecode(ops: list[LROp]):
    print(colored("low level bytecode:", "grey"))
    for i, op in enumerate(ops):
        print(colored(f"({i}) ", "grey"), end="")
        print_lr_op(op)


def show_compiler_output(
    expr: Expr,
    sequence: BytecodeSequence,
    multimethod_deps: list[MultiMethod],
    is_recompilation: bool,
):
    if not should_show_compiler_output:
        return

    print("===== COMPILER OUTPUT =====")
    if is_recompilation:
        print("(RECOMPILATION)")
    print("INPUT EXPRESSION:", expr)

    def print_op(index, bytecode, depth):
        level = "    "
        indent = level * depth

        if depth >= 5:
            print(indent + "(WARNING: recursion depth exceeded")
            return

        def prefix():
            print(indent + f"{index}: {bytecode.op}", end="")

        def basic_print_op():
            prefix()
            print("".join(" " + str(arg) for arg in bytecode.args))

        if bytecode.op == "multimethod-dispatch":
            message, nargs, methods, jump_index_on_failure = bytecode.args
            prefix()
            print(f" {message} {nargs}")
            for matchers, jump in methods:

                def matcher_str(matcher):
                    if isinstance(matcher, ParameterAnyMatcher):
                        return "<any>"
                    elif isinstance(matcher, ParameterTypeMatcher):
                        return matcher.param_type.name
                    elif isinstance(matcher, ParameterValueMatcher):
                        return f"eq({matcher.param_value})"

                print(
                    indent
                    + level
                    + f"->{jump.index} on match: {', '.join(matcher_str(matcher) for matcher in matchers)}"
                )
            print(indent + level + f"->{jump_index_on_failure.index} on failure")
        # elif bytecode.op in ["invoke-quote", "tail-invoke-quote"]:
        #     prefix()
        #     message = bytecode.args[0]
        #     quote: QuoteMethodBody = bytecode.args[1]
        #     nargs: int = bytecode.args[2]
        #     print(f" {message} {quote.compiled_body.body} {nargs}")
        #     if quote.compiled_body.bytecode:
        #         for i, c in enumerate(quote.compiled_body.bytecode.code):
        #             print_op(i, c, depth + 1)
        #     else:
        #         print(indent + level + "<invalidated, needs recompilation>")
        # elif bytecode.op == "push-closure":
        #     basic_print_op()
        #     quote: QuoteValue = bytecode.args[0]
        #     if quote.compiled_body.bytecode:
        #         for i, c in enumerate(quote.compiled_body.bytecode.code):
        #             print_op(i, c, depth + 1)
        #     else:
        #         print(indent + level + "<invalidated, needs recompilation>")
        else:
            basic_print_op()

    print("BYTECODE:")
    for i, c in enumerate(sequence.code):
        print_op(i, c, depth=1)
    print("MULTIMETHOD DEPS:")
    for multimethod in multimethod_deps:
        print("  " + multimethod.name)
    print("=========== END ===========")
