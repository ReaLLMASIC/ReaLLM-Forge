pub struct float32 {
    sign: u1,
    bexp: u8,
    fraction: u23,
}

fn unbias_exponent(exp: u8) -> s9 {
  exp as s9 - s9:127
}

pub fn float_to_int(x: float32) -> s32 {
  let exp = unbias_exponent(x.bexp);

  // Add the implicit leading one.
  // Note that we need to add one bit to the fraction to hold it.
  let fraction = u33:1 << 23 | (x.fraction as u33);

  // Shift the result to the right if the exponent is less than 23.
  let fraction =
      if (exp as u8) < u8:23 { fraction >> (u8:23 - (exp as u8)) }
      else { fraction };

  // Shift the result to the left if the exponent is greater than 23.
  let fraction =
      if (exp as u8) > u8:23 { fraction << ((exp as u8) - u8:23) }
      else { fraction };

  let result = fraction as s32;
  let result = if x.sign { -result } else { result };
  result
}

#[test]
fn float_to_int_test() {
    // 0xbeef in float32.
    let test_input = float32 {
        sign: u1:0x0,
        bexp: u8:0x8e,
        fraction: u23:0x3eef00
    };
    assert_eq(s32:0xbeef, float_to_int(test_input));
}