def parse(fasta_file, gff3_file):
    genome = ""
    with open(fasta_file, "r") as fasta:
        for lines in fasta:
            if lines.startswith(">"):
                continue
            else:
                genome += lines.strip()

    seq_dict = {"gene" : [], "non_gene" : []}
    intergenic_seq = 0
    with open(gff3_file, "r") as gff3:
        for lines in gff3:
            if lines.startswith("#"):
                continue
            else:
                if lines.strip().split("\t")[2] == "gene":
                    gene_start = lines.strip().split("\t")[3]
                    gene_end = lines.strip().split("\t")[4]

                    seq_dict["non_gene"].append(genome[int(intergenic_seq) : int(gene_start)])
                    intergenic_seq = gene_end

                    seq_dict["gene"].append(genome[int(gene_start) : int(gene_end)])

    with open("fasta.fa", "w") as out:
        for i in range(len(seq_dict["gene"])):
            if len(seq_dict["gene"][i]) == 0:
                continue
            else:
                out.write(f"1,{seq_dict["gene"][i]}\n")

            if len(seq_dict["non_gene"][i]) == 0:
                continue
            else:
                out.write(f"0,{seq_dict["non_gene"][i]}\n")

def main():
    fasta = "data/Athaliana_447_TAIR10.fa"
    gff3 = "data/Athaliana_447_Araport11.gene.gff3"

    parse(fasta, gff3)

if __name__ == "__main__":
    main()