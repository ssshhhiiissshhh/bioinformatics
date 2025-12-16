from functions import *

def Pipeline(sequence_file):
    organisms, amino_acids, nucleotides = parse_sequence_file(sequence_file)
    dist_matrix = levenshtein_matrix(nucleotides)
    align_steps = WPGMA(dist_matrix, organisms)[1]
    blosum_new = blosum_to_nested(blosum62)

    msa_nt = multiple_sequence_alignment(nucleotides, organisms, align_steps, W)
    msa_aa = multiple_sequence_alignment(amino_acids, organisms, align_steps, blosum_new)

    msa_nt_list = []
    msa_aa_list = []
    for _, seq in msa_nt.items():
        msa_nt_list.append(seq)
    for _, seq in msa_aa.items():
        msa_aa_list.append(seq)

    cleaned_seqs_nt = remove_gapped_columns(msa_nt_list)
    cleaned_seqs_aa = remove_gapped_columns(msa_aa_list)

    k2p_dist_matrix_nt = k2p_matrix(cleaned_seqs_nt)
    jc_dist_matrix_aa = jc_matrix(cleaned_seqs_aa)

    newick_nt = WPGMA(k2p_dist_matrix_nt, organisms)[0]
    newick_aa = WPGMA(jc_dist_matrix_aa, organisms)[0]

    return newick_nt, newick_aa


if __name__ == "__main__":
    file_name = "long.txt"
    result = Pipeline(file_name)
    print(f"Филогенетическое дерево для нуклеотидной последовательности: {result[0]}")
    print(f"Филогенетическое дерево для аминокислотной последовательности: {result[1]}")

    tree = Phylo.read(io.StringIO(result[0]), "newick")
    # Отображение дерева
    fig = plt.figure(figsize=(10, 8))
    axes = fig.add_subplot(1, 1, 1)
    Phylo.draw(tree, axes=axes)