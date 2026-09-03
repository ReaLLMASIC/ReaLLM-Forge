import apfloat;

pub type E4M3 = apfloat::APFloat<u32:4, u32:3>;
// # pub struct APFloat<EXP_SZ: u32, FRACTION_SZ: u32> {...}

#[test]
fn E4M3_mul_test() {
    // E4M3 8-bit floating-point format
    // Test: 
    //      0.375 * (-1.75) = -0.65625
    //                      = -0.625    (<= We cannot represent -0.65625 in E4M3)
    
    // Source: E4M3_utility.py
    // ------------------- ++++ -------------------
    // Original_dec  |     E4M3      |    E4M3_dec
    //        0.375 -> 0 0101 100   ->        0.375     c
    //        -1.75 -> 1 0111 110   ->        -1.75     d
    //     -0.65625 -> 1 0110 010   ->       -0.625     ans

    let c = E4M3 {
        sign:       u1:0b0,
        bexp:       u4:0b0101,
        fraction:   u3:0b100
    };
    let d = E4M3 {
        sign:       u1:0b1,
        bexp:       u4:0b0111,
        fraction:   u3:0b110
    };
    let ans = E4M3 {
        sign:       u1:0b1,
        bexp:       u4:0b0110,
        fraction:   u3:0b010
    };
    assert_eq(ans, apfloat::mul(c,d));
}