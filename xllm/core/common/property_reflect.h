/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <vector>

// Lightweight, dependency-free reflection for PROPERTY-based config structs.
//
// The (global) PROPERTY macro registers each field into a per-struct table so a
// consumer can iterate every plain-data field of a struct (e.g. ModelArgs,
// ParallelArgs) without hand-enumerating them. The reflection layer itself
// depends only on <type_traits> and std containers -- no nlohmann/json, no
// pybind11 -- so it can live in core config headers. The conversion to a
// concrete representation (a pybind py::dict, a JSON object, ...) is done by a
// PropertyVisitor implemented at the boundary that actually needs it.
//
// Opt a struct in with REFLECT_PROPERTIES(<StructName>) once at the top of its
// body (see macros.h). Structs without it fall back to the `void` owner and
// register nothing, so augmenting PROPERTY is safe for every existing user.

namespace xllm {

// Sink for reflected fields. The overload set is intentionally closed to the
// plain-data field types that config structs expose; any PROPERTY field whose
// type is not one of these (raw pointers, opaque vendor handles, nlohmann::json
// mapping blobs, ...) is skipped at compile time (see kVisitable / emit) and
// never reaches a visitor.
class PropertyVisitor {
 public:
  virtual ~PropertyVisitor() = default;

  virtual void visit(const std::string& name, bool value) = 0;
  virtual void visit(const std::string& name, int32_t value) = 0;
  virtual void visit(const std::string& name, int64_t value) = 0;
  virtual void visit(const std::string& name, float value) = 0;
  virtual void visit(const std::string& name, double value) = 0;
  virtual void visit(const std::string& name, const std::string& value) = 0;
  virtual void visit(const std::string& name,
                     const std::vector<int32_t>& value) = 0;
  virtual void visit(const std::string& name,
                     const std::vector<int64_t>& value) = 0;
  virtual void visit(const std::string& name,
                     const std::vector<float>& value) = 0;
  virtual void visit(const std::string& name,
                     const std::vector<double>& value) = 0;
  virtual void visit(const std::string& name,
                     const std::vector<bool>& value) = 0;
  virtual void visit(const std::string& name,
                     const std::vector<std::string>& value) = 0;
  virtual void visit(const std::string& name,
                     const std::unordered_set<int32_t>& value) = 0;

  // An engaged optional is forwarded to the matching visit() overload above; a
  // disengaged optional lands here so the visitor can represent "unset"
  // (e.g. Python None) rather than a defaulted value.
  virtual void visit_absent(const std::string& name) = 0;
};

namespace reflect {

// int32_t/int and int64_t are addressed by their fixed-width names in the
// PropertyVisitor overload set. On xLLM's LP64 targets `int` is int32_t, so a
// PROPERTY(int, ...) field routes to visit(int32_t); assert that assumption so
// a hypothetical platform where it does not hold fails loudly here rather than
// silently dropping every `int` field.
static_assert(std::is_same_v<int, std::int32_t>,
              "PropertyVisitor assumes int == int32_t on the target platform");

// Allow-list of directly visitable field types. Kept explicit (rather than
// detected from the PropertyVisitor overload set) so implicitly-convertible
// types -- notably nlohmann::json, which has broad implicit conversions -- are
// never accidentally coerced into a visit() overload.
template <typename T>
inline constexpr bool kVisitable =
    std::is_same_v<T, bool> || std::is_same_v<T, int32_t> ||
    std::is_same_v<T, int64_t> || std::is_same_v<T, float> ||
    std::is_same_v<T, double> || std::is_same_v<T, std::string> ||
    std::is_same_v<T, std::vector<int32_t>> ||
    std::is_same_v<T, std::vector<int64_t>> ||
    std::is_same_v<T, std::vector<float>> ||
    std::is_same_v<T, std::vector<double>> ||
    std::is_same_v<T, std::vector<bool>> ||
    std::is_same_v<T, std::vector<std::string>> ||
    std::is_same_v<T, std::unordered_set<int32_t>>;

// Emit a single field value. Directly visitable types are forwarded to the
// visitor; anything else is dropped at compile time.
template <typename T>
void emit(PropertyVisitor& visitor, const std::string& name, const T& value) {
  if constexpr (kVisitable<T>) {
    visitor.visit(name, value);
  }
  // Non-visitable types are intentionally skipped.
}

// std::optional fields: forward the contained value when engaged, otherwise
// report absence. Non-visitable contained types are dropped.
template <typename U>
void emit(PropertyVisitor& visitor,
          const std::string& name,
          const std::optional<U>& value) {
  if constexpr (kVisitable<U>) {
    if (value.has_value()) {
      visitor.visit(name, *value);
    } else {
      visitor.visit_absent(name);
    }
  }
}

// Per-owner field table. Populated at static-init time by the registrations the
// PROPERTY macro emits; iterated at runtime by visit_properties().
template <typename Owner>
struct PropertyTable {
  using EmitFn = void (*)(const void*, PropertyVisitor&);
  struct Field {
    const char* name;
    EmitFn emit;
  };

  static std::vector<Field>& fields() {
    static std::vector<Field> fields;
    return fields;
  }
};

// Registration hook invoked by the PROPERTY macro. A `void` owner (a struct
// that did not opt into reflection) registers nothing.
template <typename Owner>
int register_property(const char* name,
                      void (*emit)(const void*, PropertyVisitor&)) {
  if constexpr (!std::is_void_v<Owner>) {
    PropertyTable<Owner>::fields().emplace_back(
        typename PropertyTable<Owner>::Field{name, emit});
  }
  return 0;
}

}  // namespace reflect

// Visit every reflected field of `owner` in declaration order. No-op for a
// struct that did not opt into reflection.
template <typename Owner>
void visit_properties(const Owner& owner, PropertyVisitor& visitor) {
  for (const auto& field : reflect::PropertyTable<Owner>::fields()) {
    field.emit(static_cast<const void*>(&owner), visitor);
  }
}

}  // namespace xllm

// Fallback owner for PROPERTY-bearing structs that have not opted into
// reflection. Declared at global scope so unqualified lookup inside any such
// struct resolves here; REFLECT_PROPERTIES(<Struct>) shadows it with the real
// owning type at class scope.
using xllm_property_owner_t = void;
