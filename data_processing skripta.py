import time
import re
import codecs

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def isenglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext

def validation(nquads):
    print(len([nquad for nquad in nquads if len(nquad) != 5]))
    print(len([nquad for nquad in nquads if not re.match('<.*>', nquad[0]) and not nquad[0].startswith("_:")]))
    print(len([nquad for nquad in nquads if not re.match('<.*>', nquad[1])]))
    print(len([nquad for nquad in nquads if not re.match('<.*>', nquad[2])]))
    print(len([nquad for nquad in nquads if not re.match('<.*>', nquad[3])]))
    print(len([nquad for nquad in nquads if not re.match('.\n', nquad[4])]))
    
def set_difference(nquads, fnquads):
    p = set()
    for nquad in fnquads:
        p.update([nquad[3]])
    
    s = set()
    for nquad in nquads:
        s.update([nquad[3]])
    
    l = s - p
    
    return s, p, l

# Regex to match URLs
regex = re.compile(
        r'^(?:http|ftp)s?://' # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' #domain...
        r'localhost|' #localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
        r'(?::\d+)?' # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)

# Read nquads from file and create list of nquads
nquads = list()
with open(r"C:\Users\aleks\Desktop\lov.nq", encoding='utf-8') as f:
    for line in f:
        nquad = line.split(' ')
        if nquad[2].startswith('"'):
            s = ""
            i = 2
            if not (len(nquad) == 5):
                # This means sentence is split by words, so it has to be concatenated
                while i < len(nquad) - 3:
                    s += nquad[i] + " "
                    i += 1
            s += nquad[i]
            i += 1
            new_nquad = list()
            new_nquad.append(nquad[0])
            new_nquad.append(nquad[1])
            new_nquad.append(s)
            new_nquad.append(nquad[i])
            new_nquad.append(nquad[i + 1])
            nquads.append(new_nquad)
        else:
            nquads.append(nquad)

# Some nquads are poorly formated, so they need to be fixed
for nquad in nquads:
    if not re.match("^<.*>$", nquad[3]):
        nquad[2] += nquad[3]
        nquad[3] = nquad[4]
        nquad[4] = nquad[5]
        del nquad[5]
        
for nquad in nquads:
    if len(nquad) != 5:
        nquad[0] += nquad[1]
        nquad[1] = nquad[2]
        nquad[2] = nquad[3]
        nquad[3] = nquad[4]
        nquad[4] = nquad[5]
        del nquad[5]

# Now, filter out only nquads which have strings as subject
fnquads = [nquad for nquad in nquads if nquad[2].startswith('"')]

# Filter out all nquads which have http://lov.okfn.org/dataset/lov as graph labels, because they don't contain terms
fnquads = [nquad for nquad in fnquads if nquad[3] != '<http://lov.okfn.org/dataset/lov>']

# Filter out all nquads which start with _: because they are surely not related to any term
fnquads = [nquad for nquad in fnquads if not nquad[0].startswith('_:')]

# Remove unnececary suffixes from strings
for nquad in fnquads:
    nquad[2] = re.sub('\^\^<.*>$', "", nquad[2])

# If strings are anotated by language, keep only those which are anotated with english
fnquads = [nquad for nquad in fnquads if not re.match('.*"@[a-zA-Z]*(-[a-zA-Z]*)*$', nquad[2]) or (re.match('.*"@[a-zA-Z]*(-[a-zA-Z]*)*$', nquad[2]) and (re.match('.*"@en(-[a-zA-Z]*)*$', nquad[2]) or re.match('.*"@eng(-[a-zA-Z]*)*$', nquad[2])))]
           
# Remove uneccesary language anotations
for nquad in fnquads:
    nquad[2] = re.sub('@en$', "", nquad[2])
for nquad in fnquads:
    nquad[2] = re.sub('@eng$', "", nquad[2])
for nquad in fnquads:
    nquad[2] = re.sub('@en-us$', "", nquad[2])
for nquad in fnquads:
    nquad[2] = re.sub('@en-gb$', "", nquad[2])
for nquad in fnquads:
    nquad[2] = re.sub('@en-US$', "", nquad[2])
for nquad in fnquads:
    nquad[2] = re.sub('@en-GB$', "", nquad[2])
for nquad in fnquads:
    nquad[2] = re.sub('@en-au$', "", nquad[2])

# Remove apostrophes from strings
for nquad in fnquads:
    nquad[2] = nquad[2][1:len(nquad[2]) - 1]

# Remove nquads which strings are only integer numbers
fnquads = [nquad for nquad in fnquads if not nquad[2].isnumeric()]

# Remove nquads which strings are only float numbers
fnquads = [nquad for nquad in fnquads if not isfloat(nquad[2])]


# Filter out nquads which string has no letters in it
fnquads = [nquad for nquad in fnquads if not re.match('^[^a-zA-Z]*$', nquad[2])]

# Remove nquads which strings contain other than english characters
fnquads = [nquad for nquad in fnquads if isenglish(codecs.decode(nquad[2], 'unicode_escape'))]

# Remove nquads which strings are only URLs
fnquads = [nquad for nquad in fnquads if not regex.match(nquad[2])]


# Remove nquads which strings are dates
fnquads = [nquad for nquad in fnquads if not re.match("[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]", nquad[2])]
           
good_predicates = {
        '<http://www.ordnancesurvey.co.uk/ontology/Rabbit/v1.0/Rabbit.owl#Rabbit>',
        '<http://schema.org/comment>',
        '<http://usefulinc.com/ns/doap#shortdesc>',
        '<http://www.linkedmodel.org/schema/vaem#description>',
        '<http://www.ordnancesurvey.co.uk/ontology/Rabbit/v1.0/Rabbit.owl#Definition>',
        '<http://www.w3.org/2004/02/skos/core#definition>',
        '<http://semanticscience.org/resource/comment>',
        '<http://purl.org/dc/terms/description>',
        '<http://vivoweb.org/ontology/core#description>',
        '<http://www.linkedmodel.org/schema/vaem#comment>',
        '<http://metadataregistry.org/uri/profile/rdakit/toolkitDefinition>',
        '<http://www.w3.org/2000/01/rdf-schema#comment>',
        '<http://rdfs.co/bevon/description>',
        '<http://www.w3.org/ns/prov#definition>',
        '<http://www.w3.org/2000/01/rdf-schema#coment>',
        '<http://purl.obolibrary.org/obo/IAO_0000115>',
        '<http://purl.org/dc/elements/1.1/description>',
        '<http://guava.iis.sinica.edu.tw/r4r/Definition>',
        '<http://www.w3.org/2000/01/rdf-schema#description>',
        '<http://qudt.org/schema/qudt/description>',
        '<http://purl.org/imbi/ru-meta.owl#definition>',
        '<http://www.w3.org/2002/07/owl#comment>',
        '<http://vocab.gtfs.org/terms#comment>',
        '<http://vitro.mannlib.cornell.edu/ns/vitro/0.7#descriptionAnnot>'
        }


fnquads = [nquad for nquad in fnquads if nquad[1] in good_predicates or (nquad[1] == '<http://www.w3.org/2000/01/rdf-schema#label>' and nquad[3] == '<http://vocab.deri.ie/tao>')]

# Contains spanish
fnquads = [nquad for nquad in fnquads if nquad[3] != '<http://spi-fm.uca.es/spdef/models/genericTools/wikim/1.0>']

fnquads = [nquad for nquad in fnquads if nquad[3] != '<http://purl.obolibrary.org/obo/obi.owl>' or (nquad[3] == '<http://purl.obolibrary.org/obo/obi.owl>' and nquad[1] == '<http://purl.obolibrary.org/obo/IAO_0000115>')]

fnquads = [nquad for nquad in fnquads if not nquad[2].lower().startswith('depricated, no longer used as of')]

fnquads = [nquad for nquad in fnquads if not nquad[2].lower().startswith('see also')]

fnquads = [nquad for nquad in fnquads if not nquad[2].lower().startswith('example:')]

fnquads = [nquad for nquad in fnquads if not nquad[2].lower().startswith('examples:')]

fnquads = [nquad for nquad in fnquads if nquad[3] != '<http://data.lirmm.fr/ontologies/oan>']

# Contains Italian comments only
fnquads = [nquad for nquad in fnquads if nquad[3] != '<http://purl.org/LiMo/0.1#>']
           
fnquads = [nquad for nquad in fnquads if nquad[3] != '<http://rdfs.co/juso/>' or (nquad[3] == '<http://rdfs.co/juso/>' and nquad[1] == '<http://www.w3.org/2004/02/skos/core#definition>')]
           
fnquads = [nquad for nquad in fnquads if nquad[3] != '<http://www.opengis.net/ont/geosparql>' or (nquad[3] == '<http://www.opengis.net/ont/geosparql>' and nquad[1] == '<http://www.w3.org/2004/02/skos/core#definition>')]

fnquads = [nquad for nquad in fnquads if nquad[3] != '<http://rdvocab.info/RDARelationshipsWEMI>' or (nquad[3] == '<http://rdvocab.info/RDARelationshipsWEMI>' and nquad[1] == '<http://www.w3.org/2004/02/skos/core#definition>')]
           
fnquads = [nquad for nquad in fnquads if nquad[0] != '<http://www.w3.org/ns/prov-o#>']

fnquads = [nquad for nquad in fnquads if nquad[3] != '<http://www.w3.org/2004/02/skos/core>' or (nquad[3] == '<http://www.w3.org/2004/02/skos/core>' and nquad[1] == '<http://www.w3.org/2004/02/skos/core#definition>')]

fnquads = [nquad for nquad in fnquads if nquad[3] != '<http://purl.org/spar/cito/>' or (nquad[3] == '<http://purl.org/spar/cito/>' and nquad[1] == '<http://www.w3.org/2000/01/rdf-schema#comment>')]

fnquads = [nquad for nquad in fnquads if nquad[3] != '<http://purl.org/vocab/frbr/core>' or (nquad[3] == '<http://purl.org/vocab/frbr/core>' and nquad[1] == '<http://www.w3.org/2004/02/skos/core#definition>')]

fnquads = [nquad for nquad in fnquads if nquad[3] != '<http://purl.org/biotop/biotop.owl>' or (nquad[3] == '<http://purl.org/biotop/biotop.owl>' and nquad[1] == '<http://purl.org/imbi/ru-meta.owl#definition>')]           

fnquads = [nquad for nquad in fnquads if nquad[3] != '<http://iflastandards.info/ns/fr/frbr/frbrer/>' or (nquad[3] == '<http://iflastandards.info/ns/fr/frbr/frbrer/>' and nquad[1] == '<http://www.w3.org/2004/02/skos/core#definition>')]                      
           
fnquads = [nquad for nquad in fnquads if nquad[3] != '<http://iflastandards.info/ns/fr/frad/>' or (nquad[3] == '<http://iflastandards.info/ns/fr/frad/>' and nquad[1] == '<http://www.w3.org/2004/02/skos/core#definition>')]

fnquads = [nquad for nquad in fnquads if nquad[3] != '<http://www.loc.gov/premis/rdf/v1>' or (nquad[3] == '<http://www.loc.gov/premis/rdf/v1>' and nquad[1] == '<http://www.w3.org/2004/02/skos/core#definition>')]      

# Foreign languages
fnquads = [nquad for nquad in fnquads if nquad[3] != '<http://idi.fundacionctic.org/cruzar/turismo>']                           

fnquads = [nquad for nquad in fnquads if nquad[3] != '<http://purl.org/linked-data/sdmx/2009/code>' or (nquad[3] == '<http://purl.org/linked-data/sdmx/2009/code>' and nquad[1] == '<http://www.w3.org/2004/02/skos/core#definition>')]
           
fnquads = [nquad for nquad in fnquads if nquad[3] != '<http://purl.org/vocommons/voaf>' or (nquad[3] == '<http://purl.org/vocommons/voaf>' and nquad[1] == '<http://www.w3.org/2000/01/rdf-schema#comment>')]           

# Foreign languages
fnquads = [nquad for nquad in fnquads if nquad[3] != '<http://data.lirmm.fr/ontologies/poste>']

fnquads = [nquad for nquad in fnquads if nquad[3] != '<https://w3id.org/BCI-ontology>' or (nquad[3] == '<https://w3id.org/BCI-ontology>' and nquad[1] == '<http://www.w3.org/2004/02/skos/core#definition>')]

# No useful text data           
fnquads = [nquad for nquad in fnquads if nquad[3] != '<http://dati.san.beniculturali.it/SAN/>']  

# Spanish
fnquads = [nquad for nquad in fnquads if nquad[3] != '<http://spi-fm.uca.es/spdef/models/genericTools/vmm/1.0>']

fnquads = [nquad for nquad in fnquads if nquad[3] != '<http://spi-fm.uca.es/spdef/models/genericTools/itm/1.0>']   

fnquads = [nquad for nquad in fnquads if nquad[3] != '<http://vocab.lenka.no/geo-deling>']

fnquads = [nquad for nquad in fnquads if nquad[3] != '<http://www.purl.org/net/remetca#>'] 
           
fnquads = [nquad for nquad in fnquads if nquad[3] != '<http://data.ign.fr/def/topo>'] 

fnquads = [nquad for nquad in fnquads if not  'this class is deprecated' in nquad[2].lower()] 

fnquads = [nquad for nquad in fnquads if not nquad[2].startswith('check domain.')] 

fnquads = [nquad for nquad in fnquads if not nquad[2].startswith('meaning not clear.')] 

fnquads = [nquad for nquad in fnquads if not nquad[2].startswith('check range.')]

fnquads = [nquad for nquad in fnquads if not nquad[2].startswith('see:')]

fnquads = [nquad for nquad in fnquads if not  'non in the data' in nquad[2].lower()] 

fnquads = [nquad for nquad in fnquads if not  'abstract, non instantiated in the dataset' in nquad[2].lower()] 

fnquads = [nquad for nquad in fnquads if not  r'\n\t    value type: text' in nquad[2].lower()]       

for fnquad in fnquads:
    if len(fnquad) == 5:
        for fnquad1 in fnquads:
            if fnquad != fnquad1 and fnquad[2] == fnquad1[2] and fnquad[3] == fnquad1[3] and len(fnquad1) == 5:
                fnquad1.append('double')

new_fnquads = list()
for fnquad in fnquads:
    if len(fnquad) == 5:
        new_fnquads.append(fnquad)
        
fnquads = new_fnquads

nset, fset, difference = set_difference(nquads, fnquads)

# Remove vocabulary descriptions
#fnquads = [nquads for nquads in fnquads if nquads[0] not in fset and nquads[0] + '#' not in fset]
            
nset, fset, difference1 = set_difference(nquads, fnquads)

for fnquad in fnquads:
    fnquad[2] = cleanhtml(fnquad[2])
    fnquad[2] = re.sub(r'^https?:\/\/.*[\r\n]*', '', fnquad[2])
    fnquad[2] = re.sub(r"\\.", '', fnquad[2])
    fnquad[2] = fnquad[2].replace('(', '')
    fnquad[2] = fnquad[2].replace(')', '')
    fnquad[2] = fnquad[2].replace('[', '')
    fnquad[2] = fnquad[2].replace(']', '')
    fnquad[2] = fnquad[2].replace('"', '')
    fnquad[2] = fnquad[2].replace("'", '')

# Write filtered nquads
with open(r'C:\Users\aleks\Desktop\lov_filtered.nq', 'w', encoding='utf-8') as f:
    for nquad in fnquads:
        f.write(nquad[0] + " " + nquad[1] + " " + nquad[2] + " " + nquad[3] + " " + nquad[4])