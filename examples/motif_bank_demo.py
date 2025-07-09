from genetic_bpe.motif_bank import MotifBank
import pandas as pd
from datetime import datetime
import os

def main():
    # Create output directory
    output_dir = "examples/output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize motif bank
    print("Initializing Motif Bank...")
    motif_bank = MotifBank()
    
    # Sample sequences to test
    test_sequences = [
        "UGUGAUACGUGAUA",  # Human miRNA-21
        "UAGCAGCACGUAAAUAUUGGCG",  # Human miRNA-16
        "UUAAUGCUAAUCGUGAUAGGGGU",  # Human miRNA-155
        "AGCUACAUUGUCUGCUGGGUUUC",  # Human miRNA-221
        "AGCUACAUCUGGCUACUGGGU",    # Human miRNA-222
    ]
    
    # Create a detailed report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(output_dir, f"motif_analysis_{timestamp}.txt")
    
    with open(report_file, 'w') as f:
        f.write("=== Motif Bank Analysis Report ===\n\n")
        
        # 1. Initial Statistics
        f.write("1. Initial Motif Bank Statistics:\n")
        f.write("-" * 50 + "\n")
        initial_stats = motif_bank.get_motif_statistics()
        for key, value in initial_stats.items():
            f.write(f"{key}: {value}\n")
        
        # 2. Process each sequence
        f.write("\n2. Sequence Analysis:\n")
        f.write("-" * 50 + "\n")
        
        for seq in test_sequences:
            f.write(f"\nProcessing sequence: {seq}\n")
            f.write("-" * 30 + "\n")
            
            # Find motifs
            found_motifs = motif_bank.find_motifs_in_sequence(seq)
            f.write("\nFound motifs:\n")
            f.write(found_motifs.to_string())
            f.write("\n")
        
        # 3. Updated Statistics
        f.write("\n3. Updated Motif Bank Statistics:\n")
        f.write("-" * 50 + "\n")
        updated_stats = motif_bank.get_motif_statistics()
        for key, value in updated_stats.items():
            f.write(f"{key}: {value}\n")
        
        # 4. Save motifs to CSV
        csv_path = os.path.join(output_dir, "motifs.csv")
        motif_bank.save_motifs(csv_path)
        f.write(f"\nMotifs saved to: {csv_path}\n")
        
        # 5. Export to JSON
        json_path = os.path.join(output_dir, "motifs.json")
        motif_bank.export_to_json(json_path)
        f.write(f"Motifs exported to: {json_path}\n")
    
    print(f"\nAnalysis complete! Report saved to: {report_file}")
    print(f"Motifs saved to: {csv_path}")
    print(f"JSON export saved to: {json_path}")

if __name__ == "__main__":
    main() 