import pymongo
from pymongo import MongoClient
import ase.io
import os
import sys
import shutil
from datetime import *
#import warnings
#warnings.filterwarnings('error')


def get_db():
    client=MongoClient()
    db = client.materialDB
    return db


def get_collection(db):
    collect = db.cif
    return collect


def read_file(path, cif):
    if not os.path.exists('errorFile'):
        os.mkdir('errorFile')
    if (not(os.path.isdir(path)) and (".cif" in path)):
        try:
            atoms=ase.io.read(path,store_tags=True,onduplicates='keep')
        except Exception as errorInfo:
            if (type(errorInfo) == UserWarning):
                try:
                    f = open("warningLog", "a")
                    f.write(path + str(type(errorInfo)) + str(errorInfo) + '\n')
                finally:
                    f.close()
            else:
                shutil.copy(path,  "errorFile")
                try:
                    f = open("errorLog", "a")
                    f.write(path + str(type(errorInfo)) + str(errorInfo) + '\n')
                finally:
                    f.close()
        else:
            if ('_cod_database_code' in atoms.info):
                origin = '_cod_database_code'
            elif ('_database_code_icsd' in atoms.info):
                origin = '_database_code_icsd'
            n = cif.find({'origin_id': atoms.info[origin]}).count()
            if (n == 0):
                dict = get_info(path, atoms, origin)
                cif.insert_one(dict)
    else:
        files = os.listdir(path)
        for index, file in enumerate(files):
            filePath = path + '\\' + file
            if (not(os.path.isdir(filePath)) and (".cif" in filePath)):        
                try:
                    atoms = ase.io.read(filePath,store_tags=True,onduplicates='keep')
                except Exception as errorInfo:
                    if (type(errorInfo) == UserWarning):
                        try:
                            f = open("warningLog", "a")
                            f.write(filePath + str(type(errorInfo)) + str(errorInfo) + '\n')
                        finally:
                            f.close()
                    else:
                        shutil.copy(filePath,  "errorFile")
                        try:
                            f = open("errorLog", "a")
                            f.write(filePath + str(type(errorInfo)) + str(errorInfo) + '\n')
                        finally:
                            f.close()
                else:
                    if ('_cod_database_code' in atoms.info):
                        origin = '_cod_database_code'
                    elif ('_database_code_icsd' in atoms.info):
                        origin = '_database_code_icsd'
                    n = cif.find({'origin_id': atoms.info[origin]}).count()
                    if (n == 0):
                        dict = get_info(filePath, atoms, origin)
                        cif.insert_one(dict)


def get_one(value):    
    if(type(value) == list and len(value) == 1):
        return value[0]
    else:
        return value


def get_lists(atoms, profile):
    sets=[]
    for i in range(len(atoms.info[profile[0]])):
        temp={}
        for j in range(len(profile)):
            if (profile[j] in atoms.info):
                temp[profile[j]] = get_one(atoms.info[profile[j]][i])
            else:
                temp[profile[j]] = None
        sets.append(temp)
    return sets


def get_list(atoms, profile):
    dict = {}
    for i in range(len(profile)):
        if (profile[i] in atoms.info):
            dict[profile[i]] = get_one(atoms.info[profile[i]])
        else:
            dict[profile[i]] = None
    return dict


def get_info(filename, atoms, origin):
    file = open(filename, "r")
    dict ={}
    
    dict['origin'] = origin
    dict['origin_id'] = atoms.info[origin]
    dict['content'] = file.read()
    file.close()
        
    dict['_publ_section_title'] = atoms.info['_publ_section_title']
    dict['_publ_author_name'] = atoms.info['_publ_author_name']
        
    lattice = ['_cell_length_a',
    '_cell_length_b',
    '_cell_length_c',
    '_cell_angle_alpha',
    '_cell_angle_beta',
    '_cell_angle_gamma',
    '_cell_formula_units_z',
    '_refine_ls_r_factor_all',
    '_cell_volume']
    dict['lattice'] = get_list(atoms, lattice)
    
    space_group = ['_symmetry_space_group_name_h-m',
    '_symmetry_space_group_name_hall',
    '_symmetry_cell_setting',
    '_space_group_it_number',
    '_symmetry_int_tables_number']
    dict['space_group'] = get_list(atoms, space_group)
    
    references = []
    if ('icsd' in filename):
        references = ['_citation_id',
        '_citation_journal_abbrev',
        '_citation_year',
        '_citation_journal_volume',
        '_citation_page_first',
        '_citation_page_last',
        '_citation_journal_id_astm']
        dict['references'] = get_lists(atoms, references)
    else:
        references = ['_journal_page_first',
        '_journal_page_last',
        '_journal_coden_astm',
        '_journal_name_full',        
        '_journal_paper_doi',
        '_journal_volume',
        '_journal_year']
        dict['references'] = get_list(atoms, references)

    chemical_formula = ['_chemical_name_systematic',
    '_chemical_formula_structural',
    '_chemical_name_common',
    '_chemical_formula_sum']
    dict['chemical_formula'] = get_list(atoms, chemical_formula)
    
    symmetry = []
    for index,item in enumerate(atoms.info['_symmetry_equiv_pos_as_xyz']):
        symmetry.append({'_symmetry_equiv_pos_as_xyz':item,
        '_symmetry_equiv_pos_site_id': index})
    dict['symmetry_equiv_ops'] = symmetry
    
    oxidation=['_atom_type_symbol',
    '_atom_type_oxidation_number']
    dict['atoms_oxidation'] = get_lists(atoms, oxidation)
    
    sites = ['_atom_site_label',
    '_atom_site_type_symbol',
    '_atom_site_symmetry_multiplicity',
    '_atom_site_wyckoff_symbol',
    '_atom_site_fract_x',
    '_atom_site_fract_y',
    '_atom_site_fract_z',
    '_atom_site_occupancy',
    '_atom_site_attached_hydrogens',
    '_atom_site_b_iso_or_equiv']
    dict['sites'] = get_lists(atoms, sites)
    
    Create_date = {'_audit_creation_date': None, '_audit_update_record': None}
    if ('_audit_creation_date' in atoms.info):
        Create_date['_audit_creation_date'] = datetime.strptime(atoms.info['_audit_creation_date'], '%Y/%m/%d')
    if ('_audit_update_record' in atoms.info):
        Create_date['_audit_update_record'] = datetime.strptime(atoms.info['_audit_update_record'], '%Y/%m/%d')
    dict['create_date'] = Create_date

    return dict


db = get_db();
cif = get_collection(db)
#read_file(sys.argv[1], cif)
parameterlist=[];
for i in range(1, len(sys.argv)):
    para=sys.argv[i]
    arameterlist.append(para)
print parameterlist
read_file(parameterlist[0], cif)