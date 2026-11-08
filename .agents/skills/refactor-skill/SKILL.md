---
name: refactor-skill
description: Apply practical guidance on readable code, domain modeling, modern C++, clean code, behavior-preserving refactoring, and design patterns when implementing, refactoring, or reviewing software.
---

<!-- Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Readable Design and Refactoring

Use this skill when implementing or refactoring software, improving code
structure, or reviewing design choices. Apply the smallest set of principles
that helps the current task. User instructions, repository guidance, language
standards, and correctness take precedence.

The guidance below distills ideas associated with *The Art of Readable Code*,
*Domain-Driven Design*, *Effective C++*, *More Effective C++*, *Clean Code*,
*Refactoring*, and *Design Patterns*. It is engineering guidance, not a
substitute for those books or a collection of quotations. Do not invent page
references or present paraphrases as direct quotations. For C++, interpret
older advice in the context of the project's current standard, libraries, and
toolchain.

## Make code easy to understand

- Choose names that reveal the domain meaning, intent, units, and constraints.
  Prefer one term for one concept, and avoid abbreviations unless they are
  established in the domain.
- Keep related operations and data close. Give functions one coherent purpose,
  keep their abstraction level consistent, and expose the normal path clearly.
- Simplify conditions and expressions when doing so makes intent easier to
  follow. Name a meaningful intermediate result instead of leaving readers to
  decode a long expression.
- Limit variable scope and lifetime. Initialize values near their first use,
  avoid mutable state that is not needed, and make data flow visible.
- Use comments to explain intent, constraints, tradeoffs, or surprising
  behavior. Do not use comments to compensate for misleading names or tangled
  control flow.
- Make error behavior and side effects apparent at the boundary. Prefer
  explicit types and contracts over implicit call ordering or repeated parsing.

## Model the domain deliberately

- Start with the problem's vocabulary and boundaries. Use a consistent
  ubiquitous language within each bounded context, and translate explicitly at
  context boundaries.
- Introduce domain objects when they express meaningful rules or concepts.
  Use entities for identity and lifecycle, value objects for values defined
  by their attributes, and aggregates where invariants must be maintained
  together.
- Put each invariant and state transition with its natural owner. Keep
  application services focused on coordinating use cases, domain services on
  domain operations without a natural object owner, and infrastructure on
  external concerns.
- Keep DTOs, configuration, and parameter bundles simple when they are only
  carrying data. Do not turn every record into a behavior-rich class or force
  every class into a DDD category.
- Avoid generic `Manager`, `Helper`, or `Common` concepts when a precise domain
  name can state the responsibility.

## Apply C++ practices with current language tools

- Make ownership and lifetime explicit. Prefer RAII and standard-library
  resource wrappers; use `std::unique_ptr` for sole ownership and
  `std::shared_ptr` only when shared lifetime is truly required. Use raw
  pointers and references as non-owning access only when lifetime is clear.
- Establish valid members and base subobjects during construction. Prefer
  member and base initializer lists over default construction followed by
  assignment, and list initializers in declaration order.
- Design copy and move operations intentionally. Understand whether a type
  owns resources, whether copying is valid, and whether moving preserves its
  invariants. Avoid accidental expensive copies, but do not add move machinery
  where the compiler already provides the desired behavior.
- Prefer const-correct interfaces, explicit conversions, and types that
  prevent invalid states. Use `enum class`, `nullptr`, and scoped types where
  they improve safety and clarity.
- Make single-argument converting constructors `explicit` unless implicit
  conversion is an intentional part of the type's interface. Keep operators
  unsurprising; prefer named operations when overloaded syntax would obscure
  meaning.
- Make class interfaces safe to use: initialize members, define or delete
  special member functions according to ownership semantics, use virtual
  destructors for polymorphic bases, and avoid exposing mutable internals that
  bypass invariants.
- Never rely on virtual dispatch from a constructor or destructor to reach a
  more-derived implementation. Keep virtual interfaces substitutable, and
  avoid public inheritance when the derived type cannot honor the base
  contract.
- Handle exceptions with a clear safety guarantee and at the appropriate
  boundary. Preserve RAII cleanup, avoid partially updated state, and do not
  catch failures merely to hide them or continue with an invalid object.
- For operations that can fail, preserve a valid state at minimum; use the
  strong guarantee (no observable state change on failure) when it is
  practical and part of the contract. Mark operations `noexcept` only when
  they can uphold that promise.
- Treat templates, overloads, conversions, inheritance, and operator
  overloading as interface design tools. Avoid surprising implicit behavior,
  slicing, name hiding, and inheritance where composition expresses ownership
  or reuse more clearly.
- Return values rather than references or pointers to local objects. Avoid
  exposing handles into private state when callers could outlive or mutate the
  owning object; make any necessary view's lifetime explicit.
- Prefer standard containers and algorithms when they communicate intent.
  Consider allocation, copying, indirection, and locality for hot paths, but
  measure before claiming a performance benefit or complicating a simple
  design.
- Pass small cheap-to-copy values by value, large read-only inputs by const
  reference or a suitable view, and retained sink arguments by value then
  move. Choose by the actual type and lifetime contract, not a blanket rule.
- Do not mechanically copy older idioms. Use current project guidance and
  language facilities instead of obsolete constructs such as `auto_ptr`,
  owning raw pointers, or manual resource management.

## Keep code clean and changeable

- Reduce duplication when the same knowledge or rule must otherwise be
  changed in multiple places. Keep distinct policies separate even when their
  code currently looks similar.
- Prefer cohesive functions, modules, and classes with clear inputs,
  outputs, dependencies, and side effects. Do not split code into tiny helpers
  that merely add indirection without naming a useful concept.
- Make dependencies point toward stable policy. Keep I/O, frameworks, and
  storage details at explicit boundaries rather than letting them dictate the
  domain model.
- Use errors, assertions, and validation to communicate the contract at the
  correct boundary. Avoid defensive checks for conditions already guaranteed
  by a schema, constructor, or type.
- Keep tests focused on observable behavior and important invariants. Tests
  should support safe change, not encode private implementation details.

## Refactor in behavior-preserving steps

1. Identify the current behavior, the design problem, and the scope of the
   change. Inspect callers, ownership, state transitions, and error contracts.
2. When behavior is nontrivial, establish relevant tests or other executable
   evidence before restructuring. Record externally observable behavior and
   edge cases that must remain stable.
3. Make small transformations with one clear purpose: improve a name, extract
   a cohesive operation, move a rule to its owner, or introduce a boundary.
   Keep each step easy to inspect and reverse.
4. Run the narrowest useful checks after meaningful steps, then run the
   repository-required validation for the final change. Reinspect the diff for
   accidental behavior changes and unrelated edits.
5. Separate structural refactoring from intentional behavior changes where
   practical. If behavior must change, state the new contract and verify it
   directly.

Look for duplication, long or mixed-responsibility functions, misplaced
responsibilities, excessive coupling, primitive values carrying domain
meaning, and repeated conditional logic. Treat these as prompts to investigate,
not automatic commands to add abstractions.

## Choose design patterns only for real variation

- Begin with the responsibility or change pressure, not a catalog entry.
  Explain what varies, who owns that variation, and why the current design
  makes it difficult to change.
- Use composition by default for assembling capabilities and data. Use
  inheritance when substitutability is part of the contract, not merely to
  share implementation.
- Consider factories or registries for controlled construction and selection,
  strategies for interchangeable algorithms, adapters for incompatible
  interfaces, and facades for a simpler stable entry point—only when they
  solve an existing problem.
- Keep abstractions aligned with real clients and boundaries. Avoid speculative
  extension points, pattern-shaped wrappers, unnecessary virtual dispatch,
  shared ownership, or layers that only forward calls.
- Check that a pattern leaves construction, ownership, error handling, and
  domain rules easier to understand than before.

## Completion checks

Before finishing, ask:

- Can a reader understand the names, normal flow, and failure behavior without
  reconstructing hidden conventions?
- Does each domain rule have a clear owner, and are context boundaries
  explicit where needed?
- Are ownership, lifetime, copy/move behavior, and object invariants safe and
  visible in C++ interfaces?
- Did the refactor preserve existing behavior except for changes explicitly
  requested?
- Does each new abstraction or pattern solve a concrete responsibility or
  variation problem?
- Did validation exercise meaningful behavior, and are any unverified claims
  clearly identified?
