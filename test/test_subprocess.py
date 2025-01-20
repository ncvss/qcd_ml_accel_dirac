import subprocess

cpucap = subprocess.run(["lscpu"], capture_output=True, text=True)
# print(cpucap.stdout)
# print("\n")
# print(str(cpucap))
# print("\n")
avxc = False
avx2c = False
fmac = False

for l in cpucap.stdout.split("\n"):
    if l[0:5] == "Flags":
        print(sorted(l.split()))
        capflags = l.split()
        avxc = "avx" in capflags
        avx2c = "avx2" in capflags
        fmac = "fma" in capflags
        print("flexpriority" in capflags)

print(avxc, avx2c, fmac)

print()
print("Now over to the gcc command:")
gccflags = subprocess.run(["gcc","-v", "-xc", "/dev/null", "-O3", "-march=native", "-E"])
