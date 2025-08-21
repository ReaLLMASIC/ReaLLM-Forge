pub struct float<EXP_SZ: u32, FRACTION_SZ: u32> {
  sign: u1,
  bexp: uN[EXP_SZ],
  fraction: uN[FRACTION_SZ],
}

fn bias_scaler<N: u32, WIDE_N: u32 = {N + u32:1}>() -> sN[WIDE_N] {
  (sN[WIDE_N]:1 << (N - u32:1)) - sN[WIDE_N]:1
}

fn unbias_exponent<EXP_SZ: u32, SIGNED_EXP_SZ: u32 = {EXP_SZ + u32:1}>(
    exp: uN[EXP_SZ]) -> sN[SIGNED_EXP_SZ] {
  exp as sN[SIGNED_EXP_SZ] - bias_scaler<EXP_SZ>()
}

pub fn float_to_int<
    EXP_SZ: u32, FRACTION_SZ: u32, RESULT_SZ: u32,
    WIDE_EXP_SZ: u32 = {EXP_SZ + u32:1},
    WIDE_FRACTION_SZ: u32 = {FRACTION_SZ + u32:1}>(
    x: float<EXP_SZ, FRACTION_SZ>) -> sN[RESULT_SZ] {
  let exp = unbias_exponent(x.bexp);

  let fraction = uN[WIDE_FRACTION_SZ]:1 << FRACTION_SZ |
      (x.fraction as uN[WIDE_FRACTION_SZ]);

  let fraction =
      if (exp as u32) < FRACTION_SZ { fraction >> (FRACTION_SZ - (exp as u32)) }
      else { fraction };

  let fraction =
      if (exp as u32) > FRACTION_SZ { fraction << ((exp as u32) - FRACTION_SZ) }
      else { fraction };

  let result = fraction as sN[RESULT_SZ];
  let result = if x.sign { -result } else { result };
  result
}


#[test]
fn float_to_int_test() {
    // 0xbeef in float32.
    let test_input = float<8, 23> {
        sign: u1:0x0,
        bexp: u8:0x8e,
        fraction: u23:0x3eef00
    };
    assert_eq(s32:0xbeef, float_to_int<u32:8, u32:23, u32:32>(test_input));
}