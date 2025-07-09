import unittest
import os
from datetime import datetime
from test_motif_bank import TestMotifBank
from test_api import TokenizerAPITester

def run_tests():
    # Create reports directory
    reports_dir = "tests/reports"
    os.makedirs(reports_dir, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(reports_dir, f"test_report_{timestamp}.txt")
    
    with open(report_file, 'w') as f:
        f.write("=== GeneticBPE Test Report ===\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Run unit tests
        f.write("1. Running Unit Tests\n")
        f.write("-" * 50 + "\n")
        suite = unittest.TestLoader().loadTestsFromTestCase(TestMotifBank)
        runner = unittest.TextTestRunner(stream=f)
        runner.run(suite)
        
        # Run API tests
        f.write("\n2. Running API Tests\n")
        f.write("-" * 50 + "\n")
        api_tester = TokenizerAPITester()
        api_log = api_tester.test_tokenization()
        
        # Read and append API test results
        with open(api_log, 'r') as api_f:
            f.write(api_f.read())
        
        # Clean up temporary files
        os.remove(api_log)
    
    print(f"Test report generated: {report_file}")
    return report_file

if __name__ == "__main__":
    run_tests() 