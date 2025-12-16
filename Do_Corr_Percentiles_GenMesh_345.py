#!/usr/bin/env python
import os
import sys
import time
import gc
import logging
import multiprocessing
import numpy as np
import pandas as pd
import simnibs

def rankdata_average(data):
    data = np.asarray(data, dtype=np.float32)
    sorter = np.argsort(data, axis=1)
    inv = np.empty_like(sorter, dtype=np.int32)
    ranks = np.arange(data.shape[1], dtype=np.int32)
    np.put_along_axis(inv, sorter, ranks, axis=1)
    dc = np.take_along_axis(data, sorter, axis=1)
    res = np.diff(dc, axis=1) != 0
    obs = np.c_[np.ones(dc.shape[0], dtype=bool), res]
    dense = np.apply_along_axis(np.cumsum, 1, obs)
    dense = np.take_along_axis(dense, inv, axis=1)
    len_r = obs.shape[1]
    nonzero = np.count_nonzero(obs, axis=1)
    
    ranks_list = []
    unique_nonzero = np.unique(nonzero)
    for nz in unique_nonzero:
        indices = np.where(nonzero == nz)[0]
        sub_dense = dense[indices]
        nz_indices = np.array([np.flatnonzero(row) for row in obs[indices]], dtype=np.int32)
        _count = np.array([np.concatenate([row, [len_r]]) for row in nz_indices], dtype=np.int32)
        _result = 0.5 * (np.take_along_axis(_count, sub_dense, axis=1) +
                         np.take_along_axis(_count, sub_dense - 1, axis=1) + 1)
        ranks_list.append((indices, _result))
    final_ranks = np.zeros_like(data, dtype=np.float32)
    for indices, result in ranks_list:
        final_ranks[indices] = result
    return final_ranks

def compute_corr(x, y):
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    mx = x.mean(axis=-1, keepdims=True, dtype=np.float32)
    my = y.mean(axis=-1, keepdims=True, dtype=np.float32)
    xm = x - mx
    ym = y - my
    r_num = np.sum(xm * ym, axis=-1, dtype=np.float32)
    r_den = np.sqrt(np.sum(xm * xm, axis=-1, dtype=np.float32) *
                    np.sum(ym * ym, axis=-1, dtype=np.float32))
    with np.errstate(divide='ignore', invalid="ignore"):
        r = np.divide(r_num, r_den)
    return r

def spearman_row(npyMatrix, effectSize):
    npyMatrix = np.asarray(npyMatrix, dtype=np.float32)
    effectSize = np.asarray(effectSize, dtype=np.float32)
    rx = rankdata_average(npyMatrix)
    ry = rankdata_average(effectSize[None, :])
    return compute_corr(rx, ry)

def runCorrelation(npyMatrix, effectSize, corrType):
    allCoeffs = spearman_row(npyMatrix, effectSize)
    allNegLogP = 1  # Placeholder
    return allCoeffs, allNegLogP

def worker(task):
    try:
        start_time = time.time()
        currMatrix, currEffectSize, whichCorrelation, i = task
        shuffled_effect_size = np.random.permutation(currEffectSize)  
        currCoeffs, _ = runCorrelation(currMatrix, shuffled_effect_size, whichCorrelation)  
    except Exception as e:
        logging.error(f"Error processing task {i}: {e}", exc_info=True)
        return np.zeros((currMatrix.shape[0], 1), dtype=np.float32)
    return currCoeffs.reshape(-1, 1)

def parallel_process(currMatrix, currEffectSize, whichCorrelation, nPermutations, batchSize, saveToPath, n_cores):
    tasks = [(currMatrix, currEffectSize, whichCorrelation, i) for i in range(nPermutations)]
    num_batches = (nPermutations + batchSize - 1) // batchSize

    sample_result = worker(tasks[0])
    num_rows = sample_result.shape[0]
    num_cols = nPermutations
    mmap_file = np.lib.format.open_memmap(
        saveToPath,
        mode='w+',
        dtype=np.float32,
        shape=(num_rows, num_cols)
    )

    with multiprocessing.Pool(n_cores) as pool:
        for i in range(num_batches):
            batch_start = i * batchSize
            batch_end = min((i + 1) * batchSize, nPermutations)
            batch = tasks[batch_start:batch_end]
            batch_results = pool.map(worker, batch)
            batch_results_array = np.concatenate(batch_results, axis=1)
            mmap_file[:, batch_start:batch_end] = batch_results_array
            mmap_file.flush()
            del batch_results
            del batch_results_array
            gc.collect()
            print(f'Batch {i + 1} completed and saved to disk')

    pool.close()
    pool.join()
    del mmap_file
    gc.collect()
    return

def read_mmap_file_and_compute_pvalues(mmap_file_path, original_values, batchSize):
    mmap_file = np.load(mmap_file_path, mmap_mode='r')
    num_rows, num_cols = mmap_file.shape
    p_values = np.empty(num_rows, dtype=np.float32)
    for start_row in range(0, num_rows, batchSize):
        end_row = min(start_row + batchSize, num_rows)
        batch = mmap_file[start_row:end_row, :]
        print(f'Processing rows {start_row} to {end_row}')
        for i in range(batch.shape[0]):
            orig_value = original_values[start_row + i]
            p_value = np.sum(batch[i, :] >= orig_value) / num_cols
            p_values[start_row + i] = p_value
        del batch
        gc.collect()
    del mmap_file
    gc.collect()
    return p_values

def computeMesh(mesh_head, fields, writePath, variant):
    if variant == "base":
        gray_matter = mesh_head.crop_mesh(2)  
        for field_name, field_values in fields.items():
            field_flipped = field_values * 1
            M = gray_matter.elm2node_matrix()
            field_nodal = M.dot(field_flipped)
            field_data = simnibs.NodeData(field_nodal, name=field_name)
            gray_matter.add_node_field(field_data, '-' + field_name)
        gray_matter.write(writePath)
    else:
        np.save(writePath, fields)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Combined processing for correlation, percentiles, and mesh generation")
    parser.add_argument("subpath", help="Path to the subject's mesh directory.")
    parser.add_argument("data_filepath", help="Path to the CSV file.")
    args = parser.parse_args()

    logging.basicConfig(filename='error_log.log',
                        level=logging.DEBUG,
                        format='%(asctime)s:%(levelname)s:%(message)s')

    base_path = os.path.join(args.subpath, 'allMeshes')
    correlationsPath = os.path.join(args.subpath, 'correlations')
    os.makedirs(correlationsPath, exist_ok=True)

    subject_basename = os.path.basename(os.path.normpath(args.subpath))
    subject_name = subject_basename.split('m2m_')[-1]
    new_save_base = os.path.join(r"C:\Users\Gabma\OneDrive\Dokumente\tDCS_PEC_Python\HeadMeshes", subject_basename)

    df = pd.read_csv(args.data_filepath)
    expName = df['Name'].to_numpy()
    effectSize = df['fixedG'].to_numpy(dtype=np.float32)
    attributeType = df['Type'].to_numpy()
    data = pd.DataFrame({'Name': expName, 'fixedG': effectSize, 'Type': attributeType})
    result = data.groupby('Name').agg({'fixedG': 'mean', 'Type': 'first'}).reset_index()
    expName = result['Name'].to_numpy()
    effectSize = result['fixedG'].to_numpy(dtype=np.float32)
    attributeType = result['Type'].to_numpy()

    whichCorrelation = 'SpearmanRow'
    listAttributeTypes = ['ToM', 'Altruism', 'Empathy']
    doPermutations = 1
    nPermutations = 5000
    permBatchSize = 1024
    nCores = 20
    percentileBatchSize = 100000

    for currType in listAttributeTypes:
        print("Processing attribute:", currType)
        saveToPath = os.path.join(correlationsPath, currType)
        os.makedirs(saveToPath, exist_ok=True)

        type_path = os.path.join(base_path, currType)
        # for variant in ['fsavg_overlays', 'base', 'subject_overlays']:
        # for variant in ['fsavg_overlays', 'subject_overlays']:
        for variant in ['base']:
            print("Processing variant:", variant)
            if variant == "base":
                subdirs = [d for d in os.listdir(type_path) if os.path.isdir(os.path.join(type_path, d))]
                if not subdirs:
                    print("No subdirectories found in", type_path)
                    continue
                first_subdir = subdirs[0]
                currMeshHead = os.path.join(type_path, first_subdir, f'{subject_name}_TDCS_1_scalar.msh')
                if not os.path.exists(currMeshHead):
                    print("Mesh file not found:", currMeshHead)
                    continue
                mesh_head = simnibs.read_msh(currMeshHead)
            else:
                mesh_head = 0

            matrice_totale_path = os.path.join(base_path, currType, f'{currType}_matrice_totale_{variant}.npy')
            currMatrix = np.load(matrice_totale_path).astype(np.float32)
            attr_loc = (attributeType == currType)
            currEffectSize = effectSize[attr_loc]

            start_time = time.time()
            allCoeffs, _ = runCorrelation(currMatrix, currEffectSize, whichCorrelation)
            elapsed_time = time.time() - start_time
            print("Correlation time:", elapsed_time)
            corr_save_path = os.path.join(saveToPath, f'corr{whichCorrelation}_{variant}.npy')
            np.save(corr_save_path, allCoeffs)

            if doPermutations == 1:
                randCorr_path = os.path.join(saveToPath, f'randCorr{whichCorrelation}_{variant}.npy')
                parallel_process(currMatrix, currEffectSize, whichCorrelation, nPermutations, permBatchSize, randCorr_path, nCores)

                original_values = np.load(corr_save_path)
                p_values = read_mmap_file_and_compute_pvalues(randCorr_path, original_values, percentileBatchSize)
                neg_log10_p_values = -np.log10(np.clip(p_values, 1e-10, None))
                negLog_save_path = os.path.join(saveToPath, f'{currType}_{variant}_negLog10Pvalues.npy')
                np.save(negLog_save_path, neg_log10_p_values)
            else:
                negLog_save_path = None

            pec = np.load(corr_save_path)
            if negLog_save_path:
                neg_log10_p_values = np.load(negLog_save_path)
            else:
                neg_log10_p_values = None
            average_Mesh = np.mean(currMatrix, axis=1)
            fields = {'PEC': pec, 'negLog10Pvalues': neg_log10_p_values, 'averageMesh': average_Mesh}
            result_mesh_dir = os.path.join(new_save_base, 'allMeshes', 'ResultMesh', currType)
            os.makedirs(result_mesh_dir, exist_ok=True)
            writePath = os.path.join(result_mesh_dir, f'{currType}_{variant}_result_mesh.msh')
            computeMesh(mesh_head, fields, writePath, variant)
            if doPermutations == 1 and os.path.exists(randCorr_path):
                os.remove(randCorr_path)
            print("Completed variant:", variant)

        print("Completed attribute:", currType)

    print("Processing completed.")
