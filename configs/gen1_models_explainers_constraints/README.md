# additional fixes

number of iterations is unfortunately hardcoded into the
scf_bench/config/config.py

for full reproducability, set number of iterations on 500 on the genetic model.
(on some of the "later" gen1 runs on RF this was unfortunately missed)

we will set iterations on the yaml config ASAP for a better config flow.
