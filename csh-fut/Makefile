test_csh: test_csh.in csh.fut
	futhark opencl csh.fut
	cat test_csh.in | ./csh -t /dev/stderr -r 5 > /dev/null
test_csh.in:
	futhark dataset -b --u8-bounds=0:255 -g [1920][800][3]u8 -g [1920][800][3]u8 -g 5i64 -g 5i64 > test_csh.in