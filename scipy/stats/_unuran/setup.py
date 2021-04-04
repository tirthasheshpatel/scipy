import pathlib
import os
from itertools import chain

def _unuran_pre_build_hook(build_clib, build_info):
    from scipy._build_utils.compiler_helper import get_c_std_flag
    c_flag = get_c_std_flag(build_clib.compiler)
    if c_flag is not None:
        if 'extra_compiler_args' not in build_info:
            build_info['extra_compiler_args'] = []
        build_info['extra_compiler_args'].append(c_flag)

def _get_sources(dirs):
    sources = []
    for dir in dirs:
        files = os.listdir(dir)
        path = [str(dir / file) for file in files]
        sources += [source for source in path if (source.endswith(".c"))]
    return sources

def _get_version(configure_dot_ac, target_name):
    configure_dot_ac = pathlib.Path(__file__).parent / configure_dot_ac
    with open(configure_dot_ac, 'r') as f:
        s = f.read()
        start_idx = s.find(target_name)
        end_idx = s[start_idx:].find(")") + len(s[:start_idx])
        version = s[start_idx:end_idx].split(",")[1][1:-1]
    return version

def configuration(parent_package='', top_path=None):
    import numpy as np
    from numpy.distutils.misc_util import Configuration
    config = Configuration('_unuran', parent_package, top_path)

    # UNU.RAN info
    UNURAN_VERSION = _get_version("src/configure.ac", "AM_INIT_AUTOMAKE")
    UNURAN_DIR = pathlib.Path(__file__).parent.resolve()

    DEFINE_MACROS = [
        ('HAVE_ALARM', '1'),
        ('HAVE_DECL_ALARM', '1'),
        ('HAVE_DECL_GETOPT', '1'),
        ('HAVE_DECL_HUGE_VAL', '1'),
        ('HAVE_DECL_INFINITY', '1'),
        ('HAVE_DECL_ISFINITE', '1'),
        ('HAVE_DECL_ISINF', '1'),
        ('HAVE_DECL_ISNAN', '1'),
        ('HAVE_DECL_LOG1P', '1'),
        ('HAVE_DECL_SIGNAL', '1'),
        ('HAVE_DECL_SNPRINTF', '1'),
        ('HAVE_DECL_VSNPRINTF', '1'),
        ('HAVE_DLFCN_H', '1'),
        ('HAVE_FLOAT_H', '1'),
        ('HAVE_FLOOR', '1'),
        ('HAVE_GETTIMEOFDAY', '1'),
        ('HAVE_IEEE_COMPARISONS', '1'),
        ('HAVE_INTTYPES_H', '1'),
        ('HAVE_LIBM', '1'),
        ('HAVE_LIMITS_H', '1'),
        ('HAVE_MEMORY_H', '1'),
        ('HAVE_POW', '1'),
        ('HAVE_SIGNAL', '1'),
        ('HAVE_SQRT', '1'),
        ('HAVE_STDINT_H', '1'),
        ('HAVE_STDLIB_H', '1'),
        ('HAVE_STRCASECMP', '1'),
        ('HAVE_STRCHR', '1'),
        ('HAVE_STRINGS_H', '1'),
        ('HAVE_STRING_H', '1'),
        ('HAVE_STRTOL', '1'),
        ('HAVE_STRTOUL', '1'),
        ('HAVE_SYS_STAT_H', '1'),
        ('HAVE_SYS_TIME_H', '1'),
        ('HAVE_SYS_TYPES_H', '1'),
        ('HAVE_UNISTD_H', '1'),
        ('LT_OBJDIR', '".libs/"'),
        ('PACKAGE', '"unuran"'),
        ('PACKAGE_BUGREPORT', '"unuran@statmath.wu.ac.at"'),
        ('PACKAGE_NAME', '"unuran"'),
        ('PACKAGE_STRING', '"unuran %s"' % UNURAN_VERSION),
        ('PACKAGE_TARNAME', '"unuran"'),
        ('PACKAGE_URL', '""'),
        ('PACKAGE_VERSION', '"%s"' % UNURAN_VERSION),
        ('STDC_HEADERS', '1'),
        ('TIME_WITH_SYS_TIME', '1'),
        ('UNUR_ENABLE_INFO', '1'),
        ('VERSION', '"%s"' % UNURAN_VERSION),
        ('HAVE_CONFIG_H', '1')
        # ('NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION', None),
    ]

    UNURAN_DIRS = ["src/",
                   "src/src/",
                   "src/src/distr/",
                   "src/src/distributions/",
                   "src/src/methods/",
                   "src/src/parser/",
                   "src/src/specfunct/",
                   "src/src/uniform/",
                   "src/src/urng/",
                   "src/src/utils/",
                   "src/src/tests/"]
    UNURAN_DIRS_ = [UNURAN_DIR / dir_ for dir_ in UNURAN_DIRS[2:]]

    # Compile UNU.RAN as a static library:
    sources = _get_sources(UNURAN_DIRS_)

    config.add_library(
        'unuran',
        sources=sources,
        include_dirs=UNURAN_DIRS,
        libraries=['m'],
        language='c',
        macros=DEFINE_MACROS,
        _pre_build_hook=_unuran_pre_build_hook
    )

    config.add_extension(
        '_unuran_wrapper',
        sources=['_unuran_wrapper.c'],
        libraries=['unuran', 'm'],
        include_dirs=["src/",
                      "src/src/",
                      "src/src/distr/",
                      "src/src/distributions/",
                      "src/src/methods/",
                      "src/src/parser/",
                      "src/src/specfunct/",
                      "src/src/uniform/",
                      "src/src/urng/",
                      "src/src/utils/",
                      "src/src/tests/"],
        language='c'
    )

    # config.add_data_files(os.path.join('src', '*.pxd'))

    return config


if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
