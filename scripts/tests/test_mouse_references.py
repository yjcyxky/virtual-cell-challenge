from pathlib import Path
import sys
import unittest
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'dossier'))
from prepare_mouse_references import one_to_one_orthologs,mouse_symbol_index


class MouseReferenceTests(unittest.TestCase):
    def test_paralogs_and_cross_group_conflicts_are_not_forced(self):
        table=pd.DataFrame([
            ('a','9606','H1'),('a','10090','M1'),
            ('b','9606','H2'),('b','10090','M2'),('b','10090','M3'),
            ('c','9606','H3'),('c','10090','M4'),
            ('d','9606','H3'),('d','10090','M5'),
            ('e','9606','H4'),('e','10090','M6'),
            ('f','9606','H5'),('f','10090','M6')],columns=['DB Class Key','NCBI Taxon ID','Symbol'])
        mapping,_=one_to_one_orthologs(table)
        self.assertEqual(mapping,{'H1':'M1'})

    def test_case_alignment_requires_unique_mouse_symbol(self):
        table=pd.DataFrame({'NCBI Taxon ID':['10090','10090','10090','9606'],'Symbol':['Abc','Xy','XY','OTHER']})
        symbols,case=mouse_symbol_index(table)
        self.assertEqual(case,{'ABC':'Abc'})
        self.assertNotIn('OTHER',symbols)


if __name__=='__main__':unittest.main()
