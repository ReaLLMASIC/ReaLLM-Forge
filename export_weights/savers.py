import pickle

def save_to_pkl(to_save, file_name):
    with open(f"{file_name}.pkl", "wb") as f:
        pickle.dump(to_save, f)

def save_to_pte(et_program, file_name):
    with open(f"{file_name}.pte", "wb") as f:
        f.write(et_program.buffer)
