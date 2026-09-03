#!/bin/bash

# Check if the first argument is provided and is a .cc file
if [ -z "$1" ] || [[ "$1" != *.cc ]]; then
    echo "Error: Please provide a .cc file as the first parameter."
    exit 1
fi

# Extract the directory name and file name
DIR_NAME=$(dirname "$1")
FILE_NAME=$(basename "$1" .cc)

# Set the environment variables
export DIR_NAME
export FILE_NAME
export BAZEL_BIN="./Google-XLS/bazel-bin"

# echo " DIR_Name = ${DIR_NAME}"
# echo "FILE_Name = ${FILE_NAME}"
# echo "------------------------------------------------------------"

# Check if the second argument is provided
if [ -z "$2" ]; then
    echo "Error: Please provide 'build' or 'clean' as the second parameter."
    exit 1
fi

# Proceed based on the second parameter
if [ "$2" == "build" ]; then
    echo "Building..."

    # Run the commands
    ${BAZEL_BIN}/xls/contrib/xlscc/xlscc "${DIR_NAME}/${FILE_NAME}.cc" > "${DIR_NAME}/${FILE_NAME}.ir"
    ${BAZEL_BIN}/xls/tools/opt_main "${DIR_NAME}/${FILE_NAME}.ir" > "${DIR_NAME}/${FILE_NAME}.opt.ir"

    ${BAZEL_BIN}/xls/tools/codegen_main "${DIR_NAME}/${FILE_NAME}.opt.ir" \
        --generator=pipeline \
        --delay_model="asap7" \
        --output_verilog_path="${DIR_NAME}/${FILE_NAME}_pipeline.v" \
        --module_name=${FILE_NAME}_pipeline \
        --top=Run \
        --pipeline_stages=5 \
        --flop_inputs=true \
        --flop_outputs=true

    ${BAZEL_BIN}/xls/tools/codegen_main "${DIR_NAME}/${FILE_NAME}.opt.ir" \
        --generator=combinational \
        --delay_model="unit" \
        --output_verilog_path="${DIR_NAME}/${FILE_NAME}_comb.v" \
        --module_name=${FILE_NAME}_comb \
        --top=Run

    # Display the contents of the generated Verilog file
    # cat "${DIR_NAME}/${FILE_NAME}_pipeline.v"
    # cat "${DIR_NAME}/${FILE_NAME}_comb.v"

    echo "Building Completed!!!"

elif [ "$2" == "clean" ]; then
    rm "${DIR_NAME}/${FILE_NAME}.ir" "${DIR_NAME}/${FILE_NAME}.opt.ir" ${DIR_NAME}/${FILE_NAME}*.v
    echo "Cleaning Completed!!!"

else
    echo "Error: Unknown command '$2'. Use 'build' or 'clean'."
    exit 1
fi

unset DIR_NAME
unset FILE_NAME
unset BAZEL_BIN