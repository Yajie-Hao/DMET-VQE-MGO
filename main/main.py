import os

# 设置线程数
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import json
from tangelo import SecondQuantizedMolecule
from tangelo.problem_decomposition import DMETProblemDecomposition
from tangelo.problem_decomposition.dmet import Localization
from tangelo.algorithms import VQESolver
from tangelo.algorithms import FCISolver,CCSDSolver
import numpy as np
from tangelo.toolboxes.operators import count_qubits, FermionOperator, QubitOperator
from mindquantum.core.operators import QubitOperator as mq_QubitOperator
from mindquantum.core.gates import X
from mindquantum.core.circuit import Circuit
from mindquantum.core.operators import Hamiltonian
from mindquantum.simulator import Simulator

from mindquantum.algorithm.nisq import Transform
from mindquantum.algorithm.nisq import get_qubit_hamiltonian
from mindquantum.algorithm.nisq import uccsd_singlet_generator, uccsd_singlet_get_packed_amplitudes
from mindquantum.core.operators import TimeEvolution
from scipy.optimize import minimize
import time
from tangelo.toolboxes.molecular_computation.rdms import pad_rdms_with_frozen_orbitals_restricted
import scipy

# n_iter = 0

def dmet_func(chemical_potential, n_iter):
    print(f"The DMET loop is {len(n_iter)}")
    print("-------------------------")
    def get_dmet_list(chemical_potential, para):

        def mo(para):
            theta_1 = 0.0
            theta_2 = 0.0

            bond_1 = para[0]
            bond_2 = para[1]
            # bond_2 = 1.0
            x_H3 = bond_2 * np.cos(theta_1)
            y_H3 = bond_2 * np.sin(theta_1)
            x_H4 = x_H3 + bond_1 * np.cos(theta_2)
            y_H4 = y_H3 + bond_1 * np.sin(theta_2)

            # print("The x_H3 y_H3 x_H4 y_H4 is", x_H3, y_H3, x_H4, y_H4)
            mo_all = f"""
            H          {-1*bond_1}   0.0    0.0
            H          0.0    0.0    0.0
            H          {x_H3}    {y_H3}    0.0 
            H          {x_H4}    {y_H4}    0.0 
            """
            return mo_all

        # para = [0.73497551 3.01949615]
        # para = [1.0, 1.0]

        mol_mo = SecondQuantizedMolecule(mo(para), q=0, spin=0, basis="sto-3g")

        mo_para = para
        special_bounds = [
            (0.5, 2),  
            (0.5, 3)            
        ]


        fragment_atoms = [1]*4
        options_mo_dmet = {"molecule": mol_mo,
                            "fragment_atoms": fragment_atoms,
                            "fragment_solvers": "vqe",
                            "verbose": True,
                            "initial_chemical_potential":  chemical_potential,
                            # "mo":mo,
                            # "mo_para": mo_para,
                            # "special_bounds": special_bounds
                            }
        dmet = DMETProblemDecomposition(options_mo_dmet)
        dmet.build()
        scf_fragments = dmet._build_scf_fragments(chemical_potential)
        solver_fragment_list = []
        dummy_mol_list = []
        for i, info_fragment in enumerate(scf_fragments):

            # Unpacking the information for the selected fragment.
            mf_fragment, fock_frag_copy, mol_frag, t_list, one_ele, two_ele, fock = info_fragment   # t_int

            dummy_mol = dmet.fragment_builder(mol_frag, mf_fragment, fock,  #SecondQuantizedDMETFragment
                fock_frag_copy, t_list, one_ele, two_ele, dmet.uhf,
                dmet.fragment_frozen_orbitals[i])   #t_int

            solver_fragment = dmet.fragment_solvers[i]
            solver_options = dmet.solvers_options[i]
            system = {"molecule": dummy_mol}
            solver_fragment = VQESolver({**system, **solver_options})
            solver_fragment.build()
            
            solver_fragment_list.append(solver_fragment)
            dummy_mol_list.append(dummy_mol)
        return solver_fragment_list, dummy_mol_list, mo_para, special_bounds, dmet,fragment_atoms,mo
        
    import time
    t_time = time.time()
    number_of_electron  = 0.0
    para_list = []
    toggle = True

    file_path = './para_var.npy'
    para = [1.0, 1.0]
    n_para = len(para)
    num_para = int(-1*n_para)

    if len(n_iter) == 0:
        para = [1.0, 1.0]
    else:
        all_x = []
        loaded_array = np.load(f'{file_path}').tolist()
        temp_all_x = loaded_array.copy()
        all_x += temp_all_x
        para = all_x[num_para:].copy()


    solver_fragment_list, dummy_mol_list, mo_para, special_bounds, dmet,fragment_atoms, mo = get_dmet_list(chemical_potential, para)
    def get_current_core_energy(para, chemical_potential):

        mol_mo = SecondQuantizedMolecule(mo(para), q=0, spin=0, basis="sto-3g")

        # mo_para = para
        options_mo_dmet = {"molecule": mol_mo,
                            "fragment_atoms": fragment_atoms,
                            "initial_chemical_potential":  chemical_potential,
                            # "initial_chemical_potential":  0.0,
                            "fragment_solvers": "vqe",
                            "verbose": False,
                            }
        
        dmet_mo = DMETProblemDecomposition(options_mo_dmet)
        dmet_mo.build()


        core_energy = dmet_mo.orbitals.core_constant_energy
        
        return core_energy

    def minduqnatum_build(solver_fragment, dummy_mol):

        n_qubits = solver_fragment.get_resources()["circuit_width"]
        n_electrons = dummy_mol.n_active_electrons
        hartreefock_wfn_circuit = Circuit([X.on(i) for i in range(n_electrons)])

        ucc_fermion_ops = uccsd_singlet_generator(
            n_qubits, n_electrons, anti_hermitian=True)

        ucc_qubit_ops = Transform(ucc_fermion_ops).jordan_wigner()
        ansatz_circuit = TimeEvolution(ucc_qubit_ops.imag, 1.0).circuit
        ansatz_parameter_names = ansatz_circuit.params_name
        total_circuit = hartreefock_wfn_circuit + ansatz_circuit

        solver_fragment_ccsd = CCSDSolver(dummy_mol)
        total_energy, ccsd_single_amps, ccsd_double_amps = solver_fragment_ccsd.simulate()

        init_amplitudes_ccsd = uccsd_singlet_get_packed_amplitudes(
            ccsd_single_amps, ccsd_double_amps, n_qubits, n_electrons)
        init_amplitudes_ccsd = [init_amplitudes_ccsd[param_i] for param_i in ansatz_parameter_names]


        return ansatz_parameter_names, init_amplitudes_ccsd, total_circuit


    def get_fragment(para, chemical_potential = chemical_potential):
        mol_mo = SecondQuantizedMolecule(mo(para), q=0, spin=0, basis="sto-3g")


        options_mo_dmet = {"molecule": mol_mo,
                            "fragment_atoms": fragment_atoms,
                            "fragment_solvers": "vqe",
                            "initial_chemical_potential":  chemical_potential,
                            "verbose": False,
                            # "mo":mo,
                            # "mo_para": mo_para
                            }

        dmet_mo = DMETProblemDecomposition(options_mo_dmet)
        dmet_mo.build()
        # print("The chemical_potential is", chemical_potential)
        _,solver_fragment_list, dummy_mol_list=dmet_mo.get_resources(chemical_potential)

        return solver_fragment_list, dummy_mol_list, dmet_mo

    def get_total(para):
        solver_fragment_list, dummy_mol_list,dmet = get_fragment(para, chemical_potential)
        ansatz_parameter_names_list = dict()
        init_amplitudes_ccsd_list = dict()
        total_circuit_list = dict()
        for i in range(len(solver_fragment_list)):
            ansatz_parameter_names, init_amplitudes_ccsd, total_circuit = minduqnatum_build(solver_fragment_list[i], dummy_mol_list[i])
            ansatz_parameter_names_list[i] = ansatz_parameter_names

            init_amplitudes_ccsd_list[i] = init_amplitudes_ccsd
            total_circuit_list[i] = total_circuit
            # print(f"The len(init_amplitudes_ccsd_list[{i}]) is:", len(init_amplitudes_ccsd_list[i]))
        return solver_fragment_list, ansatz_parameter_names_list, init_amplitudes_ccsd_list, total_circuit_list


    def get_frag_x_gradient(para, p0_list, total_circuit_list, solver_fragment_list):

        def mind_get_total_para(a_list, str_list):
            def custom_sort(item):
                prefix, rest = item[0].split('_')
                if prefix.startswith('s'):
                    return (0, int(rest))
                elif prefix.startswith('d'):
                    rest_parts = rest.split('_')
                    i = int(rest_parts[0][1:]) if rest_parts[0][1:] else 0
                    j = int(rest_parts[1]) if len(rest_parts) > 1 else 0
                    return (1, i, j)

            sorted_result = sorted(zip(str_list, a_list), key=custom_sort)
            sorted_ansatz_name, sorted_para = zip(*sorted_result)
            sorted_para = np.array(sorted_para)
            total_para = dict(zip(str_list, a_list))

            return sorted_para, total_para
        

        def get_rdm_list(p0_list, total_circuit_list, solver_fragment_list):
            solver_fragment_list, dummy_mol_list, dmet = get_fragment(para, chemical_potential) ####chemical
            onerdm_temp_list = dict()
            twordm_temp_list = dict()
            for i in range(len(total_circuit_list)):
                p0, total_circuit, ansatz_parameter_names, solver_fragment = p0_list[i], total_circuit_list[i], total_circuit_list[i].params_name, solver_fragment_list[i]
                sorted_para_temp, total_para_temp = mind_get_total_para(p0, ansatz_parameter_names)
                onerdm_temp, twordm_temp = solver_fragment.get_rdm(total_para_temp, total_circuit, sorted_para_temp)
                onerdm_temp_list[i] = onerdm_temp
                twordm_temp_list[i] = twordm_temp
            # print("The onerdm_temp_list[0] is", onerdm_temp_list[0])
            return onerdm_temp_list, twordm_temp_list
            
        def get_rdm_x_emb_energy(para, onerdm_temp_list, twordm_temp_list):

            fragment_energy_x = 0.

            solver_fragment_list, dummy_mol_list, dmet = get_fragment(para, chemical_potential) ####chemical
            core_energy = np.real(get_current_core_energy(para, chemical_potential))

            for i in range(len(dummy_mol_list)):
                dummy_mol_temp = dummy_mol_list[i]
                onerdm_temp = onerdm_temp_list[i].copy()
                twordm_temp = twordm_temp_list[i].copy()
                onerdm_padded_temp, twordm_padded_temp = pad_rdms_with_frozen_orbitals_restricted(dummy_mol_temp, onerdm_temp, twordm_temp)
                fragment_energy_temp, temp_onerdm = dmet._compute_energy_restricted(dummy_mol_temp, onerdm_padded_temp, twordm_padded_temp)

                fragment_energy_x = fragment_energy_x + fragment_energy_temp
                fragment_energy_temp = 0.0

            fragment_energy_x += core_energy

            
            return fragment_energy_x
        
        def gradient(f, x, p0_list, total_circuit_list, solver_fragment_list, delta=1e-6):
            onerdm_temp_list, twordm_temp_list = get_rdm_list(p0_list, total_circuit_list, solver_fragment_list)

            n = len(x)
            x_gradinent_list = []
            # twordm_temp_list_temp = twordm_temp_list.copy()
            for i in range(n):
                x_plus = x.copy()
                x_plus[i] += delta
                x_minus = x.copy()
                x_minus[i] -= delta

                f_i_plus = f(x_plus, onerdm_temp_list, twordm_temp_list)

                f_x_minus = f(x_minus, onerdm_temp_list, twordm_temp_list)

                temp = (f_i_plus - f_x_minus) / (2 * delta)
                x_gradinent_list.append(temp.real)

            rdm_energy = f(x, onerdm_temp_list, twordm_temp_list)
            return np.real(rdm_energy), x_gradinent_list
        
        rdm_energy, x_gradinent_list = gradient(get_rdm_x_emb_energy, para, p0_list, total_circuit_list, solver_fragment_list, delta=1e-7)
        core_energy = get_current_core_energy(para, chemical_potential)
        return rdm_energy, x_gradinent_list, core_energy

    def get_x_energy_grad(para, p0_list, total_circuit_list, solver_fragment_list):
        energy_gradient = []
        rdm_energy, x_gradinent_list, core_energy = get_frag_x_gradient(para, p0_list, total_circuit_list, solver_fragment_list)

        for j in range(len(para)):
            temp = 0.0

            temp += x_gradinent_list[j]

            energy_gradient.append(temp)


        return rdm_energy, core_energy, energy_gradient

    def get_rdm_x_energy(para, all_x, total_circuit_list):
        p0_list = dict()
        temp_num = 0
        for i in range(len(total_circuit_list)):
            
            n_s = len(total_circuit_list[i].params_name)
            if i == 0:
                p0_list[i] = all_x[:n_s]
                temp_num += n_s
            else:
                p0_list[i] = all_x[temp_num:n_s+temp_num]
                temp_num += n_s
        def mind_get_total_para(a_list, str_list):
            def custom_sort(item):
                prefix, rest = item[0].split('_')
                if prefix.startswith('s'):
                    return (0, int(rest))
                elif prefix.startswith('d'):
                    rest_parts = rest.split('_')
                    i = int(rest_parts[0][1:]) if rest_parts[0][1:] else 0
                    j = int(rest_parts[1]) if len(rest_parts) > 1 else 0
                    return (1, i, j)

            sorted_result = sorted(zip(str_list, a_list), key=custom_sort)
            sorted_ansatz_name, sorted_para = zip(*sorted_result)
            sorted_para = np.array(sorted_para)
            total_para = dict(zip(str_list, a_list))

            return sorted_para, total_para
        
        number_of_electron = 0.0
        solver_fragment_list, dummy_mol_list, dmet = get_fragment(para, chemical_potential) #chemical_potential
        onerdm_temp_list = dict()
        twordm_temp_list = dict()
        
        for i in range(len(total_circuit_list)):
            p0, total_circuit, ansatz_parameter_names, solver_fragment = p0_list[i], total_circuit_list[i], total_circuit_list[i].params_name, solver_fragment_list[i]
            sorted_para_temp, total_para_temp = mind_get_total_para(p0, ansatz_parameter_names)
            onerdm_temp, twordm_temp = solver_fragment.get_rdm(total_para_temp, total_circuit, sorted_para_temp)
            onerdm_temp_list[i] = onerdm_temp
            twordm_temp_list[i] = twordm_temp
        # print("The onerdm_temp_list[0] theta is", onerdm_temp_list[0])
        fragment_energy_e_x = 0.
        for i in range(len(dummy_mol_list)):
            dummy_mol_temp = dummy_mol_list[i]
            onerdm_temp = onerdm_temp_list[i]
            twordm_temp = twordm_temp_list[i]
            onerdm_padded_temp, twordm_padded_temp = pad_rdms_with_frozen_orbitals_restricted(dummy_mol_temp, onerdm_temp, twordm_temp)
            fragment_energy_temp, temp_onerdm = dmet._compute_energy_restricted(dummy_mol_temp, onerdm_padded_temp, twordm_padded_temp)

            n_electron_frag = np.trace(temp_onerdm[: dummy_mol_list[i].t_list[0], : dummy_mol_list[i].t_list[0]])

            fragment_energy_e_x += np.real(fragment_energy_temp) 
            number_of_electron += n_electron_frag
            print("number_of_electron ", number_of_electron)
        core_self_energy = get_current_core_energy(para, chemical_potential)
        fragment_energy_e_x += core_self_energy
        ref_number_active_electrons = dmet.orbitals.number_active_electrons
        return fragment_energy_e_x, number_of_electron, ref_number_active_electrons


    def get_dmet_iter(para, all_x, temp_theta):
        print
        solver_fragment_list, dummy_mol_list,dmet = get_fragment(para, chemical_potential)
        temp_num = 0
        p0_list = dict()
        for i in range(len(total_circuit_list)):
            
            n_s = len(total_circuit_list[i].params_name)
            if i == 0:
                p0_list[i] = all_x[:n_s]
                temp_num += n_s
            else:
                p0_list[i] = all_x[temp_num:n_s+temp_num]
                temp_num += n_s

        new_p0 = []
        for f in range(len(solver_fragment_list)):
            print("Optimize fragment-{f} energy")
            hamiltonian_QubitOp = mq_QubitOperator.from_openfermion(QubitOperator.to_openfermion(solver_fragment_list[f].qubit_hamiltonian))

            n_qubits = solver_fragment_list[f].get_resources()["circuit_width"]
            n_electrons = dummy_mol_list[f].n_active_electrons

            hartreefock_wfn_circuit = Circuit([X.on(i) for i in range(n_electrons)])
            # print(hartreefock_wfn_circuit)

            ucc_fermion_ops = uccsd_singlet_generator(
                n_qubits, n_electrons, anti_hermitian=True)

            ucc_qubit_ops = Transform(ucc_fermion_ops).jordan_wigner()
            ansatz_circuit = TimeEvolution(ucc_qubit_ops.imag, 1.0).circuit
            # ansatz_parameter_names = ansatz_circuit.params_name
            total_circuit = hartreefock_wfn_circuit + ansatz_circuit
            print(total_circuit.summary())

            # from tangelo.algorithms import FCISolver, CCSDSolver
            solver_fragment_ccsd = CCSDSolver(dummy_mol_list[f])
            total_energy, ccsd_single_amps, ccsd_double_amps = solver_fragment_ccsd.simulate()
            print("The CCSD energy is:", total_energy)
            solver_fragment_fci = FCISolver(dummy_mol_list[f])
            print("The FCI energy is:", solver_fragment_fci.simulate())

            init_para = p0_list[f].copy()

            grad_ops = Simulator('mqvector_gpu', total_circuit.n_qubits).get_expectation_with_grad(
                Hamiltonian(hamiltonian_QubitOp.real),
                total_circuit)

            # import numpy as np
            import time
            t_time = time.time()
            def fun(p0, molecule_pqc, energy_list=None):
                f, g = molecule_pqc(p0)
                f = np.real(f)[0, 0]
                g = np.real(g)[0, 0]
                if energy_list is not None:
                    energy_list.append(f)
                    if len(energy_list) % 5 == 0:
                        print(f"Step: {len(energy_list)},\tenergy: {f}")
                        t1_time = time.time()
                        print("iterative time:", t1_time - t_time)
                # print("The grad list is:", g)
                return f, g
        
            
            energy_list = []

            res = minimize(fun, init_para, args=(grad_ops, energy_list), method='bfgs', jac=True, options={'maxiter': temp_theta})

            optimal_para = res.x
            # print(optimal_para)
            new_p0 += np.array(optimal_para).tolist()
            # print(new_p0)
        new_all_x = new_p0 + para
        return new_all_x

    def all_lost_grad_para(para, all_x_no_para, total_circuit_list, n_para, energy_list):
        # global toggle

        para = para
        all_x = np.array(all_x_no_para).tolist() + np.array(para).tolist()
        print("The para is (all lost grad)", para)

        all_energy = 0.0
        temp_num = 0
        p0_list = dict()
        for i in range(len(total_circuit_list)):
            
            n_s = len(total_circuit_list[i].params_name)
            if i == 0:
                p0_list[i] = all_x[:n_s]
                temp_num += n_s
            else:
                p0_list[i] = all_x[temp_num:n_s+temp_num]
                temp_num += n_s



        ###### self.toggle = False

        rdm_energy, core_energy, grad_para = get_x_energy_grad(para, p0_list, total_circuit_list, solver_fragment_list)


        core_self_energy = core_energy.copy()
        
        para_grad_norm = np.linalg.norm(grad_para)


        para_list.append(para.copy()) 

        new_grad_list = grad_para
        peanlty_energy = rdm_energy
        theta_grad_norm = 0.0

        if energy_list is not None:
            energy_list.append(rdm_energy)
            # energy_list.append(rdm_energy)
            if len(energy_list) % 1 == 0:
                print(f"Step: {len(energy_list)},\tenergy: {peanlty_energy}, \core_energy: {core_energy}, \t theta_grad_norm: {theta_grad_norm}, \t taoogle: {toggle}")
                print(f"Step: {len(energy_list)},\trdm_energy: {rdm_energy},\tallenergy: {all_energy}, \para_grad_norm: {para_grad_norm}")
                t1_time = time.time()
                print("iterative time:", t1_time - t_time)
        return peanlty_energy, new_grad_list

    file_path = './para_var.npy'
    
    n_para = len(mo_para)
    num_para = int(-1*n_para)

    if len(n_iter) == 0:
        all_x = []
        para = mo_para
        solver_fragment_list, ansatz_parameter_names_list, init_amplitudes_ccsd_list, total_circuit_list = get_total(para)
        # all_x = opt_all_x_temp
        for i in range(len(solver_fragment_list)):
            all_x += init_amplitudes_ccsd_list[i]
        all_x += para
    else:
        all_x = []
        loaded_array = np.load(f'{file_path}').tolist()
        temp_all_x = loaded_array.copy()
        all_x += temp_all_x
        para = all_x[num_para:].copy()
        solver_fragment_list, ansatz_parameter_names_list, init_amplitudes_ccsd_list, total_circuit_list = get_total(para)

    print("The all_x is:", all_x)

    energy_list = []


    temp_theta = 2
    temp_para = 2
    all_iter = 5

    bounds = special_bounds

    for i in range(all_iter):

        if (i+1) % 2 == 0:
            if (i+1) == int(all_iter-1):
                temp_para = 500
            toggle = False
            print("The iter is ", i+1, "self.toggle :", toggle)
            print("optimize para", "maxiter :", temp_para)
            all_x_no_para = all_x[:-n_para]
            res = minimize(all_lost_grad_para, para, args=(all_x_no_para, total_circuit_list, n_para, energy_list), method='L-BFGS-B', jac=True, bounds=bounds, options={'maxiter': temp_para})
            # res = my_simple_gradient_descent(all_lost_grad_para, para, args=(all_x_no_para, total_circuit_list, n_para, energy_list), learning_rate=0.1, max_iter=1000, disp=True)
            opt_x = all_x_no_para + np.array(res.x).tolist()
            para = np.array(res.x).tolist()
            all_x = opt_x
        else:
            toggle = True
            if (i+1) == all_iter or (i+1) == int(all_iter-2):
                temp_theta = 500
            print("The iter is ", i+1, "self.toggle :", toggle)
            print("optimize theta", "maxiter :", temp_theta)
            all_x = get_dmet_iter(para, all_x, temp_theta)
            rdm_energy, number_of_electron, ref_number_active_electrons = get_rdm_x_energy(para, all_x, total_circuit_list)
            # para = np.array(res.x).tolist()
            para = all_x[num_para:].copy()
            print("new_all_x is", all_x)
            print("rdm_energy is", rdm_energy)

        optimal_all_x = all_x
        np.save(f'{file_path}',optimal_all_x)
  
        print("The self.para_list is", para_list)

    n_iter.append(1.0)
    optimal_all_x = all_x
    np.save(f'{file_path}',optimal_all_x)
  
    return np.real(number_of_electron - ref_number_active_electrons)
chemical_potential = 0.0
n_iter = []
result = scipy.optimize.newton(dmet_func, chemical_potential, tol=1e-5, args=(n_iter,))
print("The optimized chemical_potential is", result.real)