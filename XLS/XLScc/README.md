# Welcome to **OpenLLM-XLS**!

## Step #0 - Clone Repo and Setup for OpenLLM-XLS:
```
curl -fsSL https://github.com/Mars-Cat2023/OpenLLM-XLS/raw/main/init.sh -o init.sh && chmod +x init.sh && ./init.sh
```

Notes:
1. [Under Construction] When you fail with `version GLIBC_2.34' not found`, upgrading your `glibc` to `v2.35` will be helpful:  
Go into the `OpenLLM-XLS` folder and run the following command:
```
# [Under Construction]
# chmod +x init_glibc_2.35.sh && ./init_glibc_2_35.sh
```

## Step #1 - Running Generation of Verilogs:
```
chmod +x generate_verilogs.sh
```
An input file: `<cc_file_name>.cc`  
**Here, we strongly recommend putting the `<cc_file_name>.cc` file in a new empty folder.**
### (1) Building (Generation) Verilogs:
```
./generate_verilogs.sh <cc_file_name>.cc build
```

### (2) Cleaning IR and Verilogs Outputs:
```
./generate_verilogs.sh <cc_file_name>.cc clean
```

------
For example, generate from our input file example `./examples/case_1/matrix_mul_10.cc`:

```
./generate_verilogs.sh ./examples/case_1/matrix_mul_10.cc build
```
and clean all the outputs
```
./generate_verilogs.sh ./examples/case_1/matrix_mul_10.cc clean
```
