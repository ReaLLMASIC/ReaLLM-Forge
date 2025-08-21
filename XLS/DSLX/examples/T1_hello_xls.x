fn hello_xls(hello_string: u8[11]) {
  trace!(hello_string);
}

#[test]
fn hello_test() {
  hello_xls("Hello, XLS!")
}

