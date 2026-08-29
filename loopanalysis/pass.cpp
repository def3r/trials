// Registration hub for the MinPass plugin (build/MinPass.so). Each of the
// five QPU-offload passes below is implemented and self-registered in its
// own <name>_pass.cpp; this file's only job is to declare and call each
// register*Pass(PassBuilder&) function so they all end up in one shared
// library. See PROJECT_INDEX.md for what each pass identifies, where its
// tests/analysis live, and the pass<->bridge coupling every one of them
// depends on.
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Plugins/PassPlugin.h"
#include "llvm/Support/Compiler.h"

using namespace llvm;

void registerMaxCutCppPass(PassBuilder& PB);
void registerTspPass(PassBuilder& PB);
void registerKColorPass(PassBuilder& PB);
void registerCliquePass(PassBuilder& PB);
void registerFactorPass(PassBuilder& PB);

extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "MinPass", LLVM_VERSION_STRING,
          [](PassBuilder& PB) {
            registerMaxCutCppPass(PB);
            registerTspPass(PB);
            registerKColorPass(PB);
            registerCliquePass(PB);
            registerFactorPass(PB);
          }};
}
