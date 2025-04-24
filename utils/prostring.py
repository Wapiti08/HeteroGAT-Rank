'''
 # @ Create Time: 2025-04-24 15:34:40
 # @ Modified time: 2025-04-24 15:34:42
 # @ Description: string processing function to reduce length by removing unnecessary part
 '''

import os
import re

def process_string(input_string, max_len=500):
    # step1: split the string by spaces
    parts = input_string.split()

    # step2: remove repeated values -- keeps the order while removing duplicates
    unique_parts = list(dict.fromkeys(parts))

    # step3: keep only the last two layers of each path and remove hash-like parts
    processed_parts = []
    for part in unique_parts:
        # remove hash-like parts
        if re.search(r'[a-fA-F0-9\-]{5,}', part):
            continue

        # if the part looks like a path
        if "/" in part:
            path_parts = part.split("/")
            # keep the last two layers
            processed_part = '/'.join(path_parts[-2:])
            processed_parts.append(processed_part)
        else:
            processed_parts.append(part)
    
    # step4: tjoin the processed parts into a string
    result_string = ''.join(processed_parts)

    # step5: trim the string to the max length if necessary
    if len(result_string) > max_len:
        result_string = result_string[:max_len]

    return result_string


# Test the function with your input string
input_string = 'rustc --crate-name build_script_build /usr/local/cargo/registry/src/github.com-1ecc6299db9ec823/crc32fast-1.3.2/build.rs --error-format=json --json=diagnostic-rendered-ansi,artifacts,future-incompat --crate-type bin --emit=dep-info,link -C embed-bitcode=no -C debuginfo=2 --cfg feature="default" --cfg feature="std" -C metadata=d495831a59ec6094 -C extra-filename=-d495831a59ec6094 --out-dir /app/target/debug/build/crc32fast-d495831a59ec6094 -L dependency=/app/target/debug/deps --cap-lints allow'
processed_string = process_string(input_string)
print(processed_string)
