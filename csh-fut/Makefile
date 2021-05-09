test_csh: test_1920_800_5i_8k_8p.in csh.fut
	futhark opencl csh.fut
	cat test_1920_800_5i_8k_8p.in | ./csh -t /dev/stderr -r 5 > /dev/null
test_1920_800_5i_8k_8p.in:
	futhark dataset -b --u8-bounds=0:255 -g [1920][800][3]u8 -g [1920][800][3]u8 -g 5i64 -g 8i64 -g 8i64 > test_1920_800_5i_8k_8p.in
benchmark: l2_data.in csh_test.fut
	futhark bench --backend=opencl csh_test.fut
l2_data.in:
	futhark dataset -b -g [1517009][23]i64 -g [1517009][23]i64 --i64-bounds=0:1517008 -g [1517009][16]i64 > l2_data.in
auto_bug.in:
	futhark dataset -b --i64-bounds=0:262143 -g [2][1517009]i64 -g [2][262144][2]i64 -g [1517009][23]i64 -g 8i64 -g 1913i64 -g 793i64 > auto_bug.in
setup_pkg:
	futhark pkg add github.com/diku-dk/sobol
	futhark pkg add github.com/diku-dk/sorts
	futhark pkg add github.com/diku-dk/cpprandom
	futhark pkg sync
clean:
	rm *.in