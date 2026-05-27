fn main() {
    let matname = std::env::args().nth(1).expect("mat file");
    let mut mat = matio::open(matname, matio::Access::Read).unwrap();
    for var in mat.vars() {
        let name = var.name().unwrap().unwrap();
        println!("name: {}", name);
        if name == "Pose_Para" {
            println!("{:?}", var.value());
        }
    }
}
