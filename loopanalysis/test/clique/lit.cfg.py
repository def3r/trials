import lit.formats
import os

config.name = "clique-pass"
config.test_format = lit.formats.ShTest(False)
config.suffixes = {'.ll'}

_dir          = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.normpath(os.path.join(_dir, '..', '..'))

_llvm_install = os.environ.get('LLVM_PROJECT_INSTALL',
                               '/home/def3r/def3r/llvm-project/install')
_llvm_build   = os.environ.get('LLVM_PROJECT_BUILD',
                               '/home/def3r/def3r/llvm-project/build')

# Only _opt.ll files are tests; skip raw (.ll) IR.
config.excludes = {
    f for f in os.listdir(_dir)
    if f.endswith('.ll') and not f.endswith('_opt.ll')
}

config.substitutions.append(('%opt',       os.path.join(_llvm_install, 'bin', 'opt')))
config.substitutions.append(('%FileCheck', os.path.join(_llvm_build,   'bin', 'FileCheck')))
config.substitutions.append(('%plugin',    os.path.join(_project_root, 'build', 'MinPass.so')))
