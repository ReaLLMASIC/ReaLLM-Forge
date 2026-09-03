import struct

def dec_to_E4M3(value):
    ''' 
        Format: dec_to_E4M3(0.375) = "0 0101 100"
    '''
    sign = 0 if value >= 0 else 1
    value = abs(value)
    
    if value == 0:
        return f"{sign} 0000 000"

    binary = struct.unpack('!I', struct.pack('!f', value))[0]

    exponent = (binary >> 23) & 0xFF
    fraction = binary & 0x7FFFFF
    
    unbiased_exponent = exponent - 127
    e4m3_exponent = unbiased_exponent + 7
    
    mantissa = (fraction >> 20) & 0x7
    
    if unbiased_exponent == -127 and fraction == 0:
        return f"{sign} 0000 000"
    
    if e4m3_exponent > 15:
        e4m3_exponent = 15
    
    if e4m3_exponent < 0:
        e4m3_exponent = 0
    
    return f"{sign} {format(e4m3_exponent, '04b')} {format(mantissa, '03b')}"


def E4M3_to_dec(byte_str):
    ''' 
        Format: E4M3_to_dec("0 0101 100") = 0.375
    '''
    sign = int(byte_str[0])
    exponent = int(byte_str[2:6], 2)
    mantissa = int(byte_str[7:], 2)
    
    unbiased_exponent = exponent - 7
    
    mantissa_value = 1 + mantissa / 8
    
    result = (-1) ** sign * mantissa_value * (2 ** unbiased_exponent)
    return result


def print_E4M3(xarray):
    print("------------------- ++++ -------------------")
    print(f"{'Original_dec':^12}  | {'E4M3':^12}  |  {'E4M3_dec':^12}")
    for x in xarray:
        print(f"{x:12} -> {dec_to_E4M3(x):12} -> {E4M3_to_dec(dec_to_E4M3(x)):12}")
    print()


if __name__ == "__main__":
    print_E4M3([1.125, 3.5, 4.625])
    print_E4M3([0.375, -1.75, -0.65625])
    pass
