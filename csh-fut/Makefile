test_csh: test_csh.in csh.fut
	futhark opencl csh.fut
	cat test_csh.in | ./csh -t /dev/stderr -r 5 > /dev/null
test_csh.in:
	futhark dataset -b --i8-bounds=-128:127 -g [3][1280][1920]i8 -g [3][1920][1280]i8 > test_csh.in