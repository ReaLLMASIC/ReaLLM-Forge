import apfloat;

pub type E4M3 = apfloat::APFloat<u32:4, u32:3>;
// # pub struct APFloat<EXP_SZ: u32, FRACTION_SZ: u32> {...}

#[test]
fn E4M3_add_test() {
    // E4M3 8-bit floating-point format
    // Test: 
    //      1.125 + 3.5 = 4.625
    //                  = 4.5           (<= We cannot represent 4.625 in E4M3)
    
    // Source: E4M3_utility.py
    // ------------------- ++++ -------------------
    // Original_dec  |     E4M3      |    E4M3_dec
    //        1.125 -> 0 0111 001   ->        1.125     a
    //          3.5 -> 0 1000 110   ->          3.5     b
    //        4.625 -> 0 1001 001   ->          4.5     ans
    
    let a = E4M3 {
        sign:       u1:0b0,
        bexp:       u4:0b0111,
        fraction:   u3:0b001
    };
    let b = E4M3 {
        sign:       u1:0b0,
        bexp:       u4:0b1000,
        fraction:   u3:0b110
    };
    let ans = E4M3 {
        sign:       u1:0b0,
        bexp:       u4:0b1001,
        fraction:   u3:0b001
    };
    assert_eq(ans, apfloat::add(a,b));
}