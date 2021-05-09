# annf
Instructions to reproduce autotune error:

navigate to /csh-fut
> make setup_pkg

> make auto_bug1.in

> futhark autotune --backend=opencl autotune_bug.fut

Output will be entries with values of 2 billion.
> futhark opencl autotune_bug.fut

> cat auto_bug.in | ./autotune_bug -t /dev/stderr -r 1 > /dev/null

It'll run
> cat auto_bug.in | ./autotune_bug -t /dev/stderr -r 1 --tuning=autotune_bug.fut.tuning > /dev/null

It'll fail with an error due to a memory allocation failure.
Alternatively it can be run with
> futhark bench --backend=opencl autotune_bug.fut

and
> futhark bench --backend=opencl --pass-option=--tuning=autotune_bug.fut.tuning autotune_bug.fut

respectively.

A tuning file can be produced on a smaller dataset, which when used on the larger(and more typical) dataset performs better.

> make auto_bug2.in

> futhark autotune --backend=opencl autotune_bug.fut

> make auto_bug1.in

> futhark bench --backend=opencl --pass-option=--tuning=autotune_bug.fut.tuning autotune_bug.fut