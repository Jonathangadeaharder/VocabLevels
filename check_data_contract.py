"""Acceptance checks for the vocab data contract of 2026-08-19.

The contract (Vidiom docs/specs/2026-08-19-vocab-data-contract.md) makes the
normalized English gloss the join key between languages and defines five
acceptance criteria for every delivery. This script runs them mechanically.

Run from repo root:
    python check_data_contract.py                # measure the CSV baseline
    python check_data_contract.py delivery/      # gate a TSV delivery

The delivery directory holds one TSV per language, UTF-8, with the header:
    lemma	pos	english_gloss	english_pos	cefr	rank	concept_key
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from vocab_schema import LEVELS

ROOT = Path(__file__).parent

LANG_DIRS = {
    "english": "en",
    "german": "de",
    "spanish": "es",
    "french": "fr",
    "swedish": "sv",
    "arabic": "ar",
    "dutch": "nl",
    "chinese": "zh",
}
CODE_TO_DIR = {code: name for name, code in LANG_DIRS.items()}

GLOSS_ASCII = re.compile(r"^[A-Za-z '.-]+$")
MIN_COVERAGE = 0.60

ARABIC_SCRIPT = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")
CHINESE_SCRIPT = re.compile(r"[\u4E00-\u9FFF\u3400-\u4DBF]")
FORBIDDEN_JUNK_LEMMAS = {
    "词",
    "复合词",
    "名词",
    "动词",
    "形容词",
    "副词",
    "介词",
    "代词",
    "连词",
    "感叹词",
    "X",
}

COGNATE_ALLOWLIST: dict[str, set[str]] = {'de': {'all', 'argument', 'arm', 'baby', 'bitter', 'butter', 'chance', 'chaos',
        'chef', 'clan', 'club', 'computer', 'design', 'extra', 'fan', 'film',
        'finger', 'fit', 'format', 'frost', 'glas', 'golf', 'gras', 'hand',
        'hey', 'hi', 'hotel', 'humor', 'ideal', 'image', 'imitation', 'in',
        'index', 'invalide', 'job', 'kanal', 'klima', 'kontakt', 'limit',
        'manager', 'maske', 'masseur', 'material', 'medium', 'mild', 'minute',
        'moment', 'motor', 'museum', 'musik', 'name', 'nation', 'nature',
        'nerv', 'nylon', 'oh', 'okay', 'onkel', 'oper', 'optik', 'organ',
        'original', 'panik', 'papier', 'park', 'partner', 'party', 'pause',
        'perfekt', 'person', 'phase', 'pilot', 'plan', 'plastik', 'podium',
        'post', 'prinz', 'problem', 'professor', 'profil', 'programm',
        'projekt', 'protest', 'prozess', 'pulver', 'punkt', 'quarz', 'radio',
        'rang', 'rate', 'reaktion', 'real', 'region', 'rekord', 'religion',
        'rest', 'ring', 'risiko', 'rolle', 'rose', 'route', 'salat', 'sand',
        'satellit', 'schock', 'sekunde', 'senat', 'senior', 'service', 'signal',
        'situation', 'so', 'sofa', 'soldat', 'solo', 'sommer', 'sport', 'spur',
        'start', 'station', 'status', 'stil', 'stopp', 'stress', 'struktur',
        'student', 'studio', 'sturm', 'substanz', 'super', 'symbol', 'system',
        'tabelle', 'talent', 'tango', 'tank', 'tarif', 'team', 'tempo',
        'tendenz', 'tennis', 'tenor', 'termin', 'test', 'text', 'theater',
        'thema', 'theorie', 'ticket', 'tiger', 'titel', 'toilette', 'toleranz',
        'ton', 'total', 'tour', 'tourist', 'tradition', 'training', 'trakt',
        'transport', 'trend', 'tribunal', 'trick', 'trio', 'tunnel',
        'turbulenz', 'turnier', 'typ', 'union', 'universum', 'vakuum', 'ventil',
        'verband', 'version', 'veteran', 'video', 'virus', 'visum', 'vokal',
        'volumen', 'vulkan', 'warm', 'wind', 'winter', 'wort', 'wow', 'zebra',
        'zement', 'zentrum', 'zirkus', 'zivil', 'zone', 'zoo', 'zucchini'},
 'es': {'acento', 'actor', 'admirable', 'adulto', 'album', 'alcohol', 'alerta',
        'alias', 'altar', 'amigo', 'animal', 'aplauso', 'area', 'argumento',
        'arte', 'artista', 'atlas', 'auto', 'autor', 'balance', 'ballet',
        'banco', 'banda', 'bar', 'base', 'bazar', 'bebe', 'blues', 'bolero',
        'bono', 'boom', 'box', 'bronce', 'cable', 'cafe', 'canal', 'cancer',
        'canon', 'caos', 'capital', 'central', 'cereal', 'ceremonia', 'cero',
        'champan', 'chance', 'chasis', 'chef', 'choque', 'cine', 'circo',
        'civil', 'clan', 'clase', 'claxon', 'clic', 'clima', 'climax', 'clip',
        'club', 'codigo', 'color', 'coma', 'comic', 'comite', 'conductor',
        'confort', 'congestion', 'congreso', 'contacto', 'control', 'cordon',
        'corte', 'cosmos', 'crater', 'credito', 'crema', 'crisis', 'crucial',
        'cuadro', 'culto', 'curva', 'danza', 'debate', 'debut', 'decada',
        'decision', 'deficit', 'deplorable', 'desastre', 'detalle', 'detective',
        'dia', 'diagonal', 'dialogo', 'dieta', 'dilema', 'dinamo', 'diploma',
        'director', 'disco', 'discurso', 'diseno', 'distrito', 'division',
        'doctor', 'dolar', 'drama', 'duo', 'eco', 'editor', 'efecto', 'ego',
        'elite', 'email', 'emblema', 'emision', 'enigma', 'era', 'erosion',
        'error', 'escala', 'escena', 'esfera', 'espacio', 'espectro', 'espia',
        'estado', 'estatua', 'estilo', 'estudio', 'etapa', 'etica', 'etiqueta',
        'evidencia', 'examen', 'exito', 'experto', 'explosion', 'factor',
        'familia', 'fan', 'farsa', 'fase', 'fatal', 'favor', 'fax', 'fe',
        'federal', 'festival', 'fibra', 'figura', 'fila', 'filtro', 'final',
        'firme', 'flauta', 'flora', 'flota', 'foco', 'fondo', 'forma',
        'formato', 'formula', 'foro', 'fosil', 'foto', 'fracaso', 'frase',
        'frecuencia', 'frontal', 'frontera', 'fuga', 'furia', 'fusion',
        'futuro', 'gala', 'galeria', 'gangster', 'garage', 'gas', 'gasolina',
        'gato', 'general', 'genio', 'germen', 'gesto', 'gigante', 'globo',
        'gloria', 'gol', 'golf', 'goma', 'gorila', 'gracia', 'grado', 'grafico',
        'gramo', 'granito', 'grava', 'grave', 'grieta', 'grifo', 'grillo',
        'grupo', 'guante', 'guardia', 'guia', 'guion', 'guitarra', 'gula',
        'gusto', 'habitat', 'habito', 'hall', 'hamburguesa', 'harem', 'hecho',
        'helicoptero', 'helio', 'hemisferio', 'heroe', 'hielo', 'hierba',
        'higiene', 'himno', 'hipotesis', 'historia', 'hobby', 'hogar', 'hola',
        'homenaje', 'hongo', 'honor', 'hora', 'horizonte', 'hormona', 'horror',
        'hospital', 'hostil', 'hotel', 'humor', 'huracan', 'icono', 'idea',
        'ideal', 'idolo', 'iglu', 'imagen', 'impacto', 'imperio', 'impulso',
        'incendio', 'indice', 'inercia', 'infante', 'infierno', 'informe',
        'ingenio', 'ingreso', 'injuria', 'inmune', 'insecto', 'insignia',
        'instinto', 'instituto', 'insulto', 'interes', 'interior', 'internet',
        'intruso', 'invasion', 'invento', 'inverso', 'ironia', 'isla', 'item',
        'jaguar', 'jardin', 'jazz', 'jeans', 'jefe', 'jersey', 'jockey', 'judo',
        'juez', 'jugo', 'juicio', 'jungla', 'junior', 'junta', 'jura',
        'justicia', 'karaoke', 'karma', 'kilo', 'kilometro', 'kiosco', 'kiwi',
        'koala', 'kung fu', 'labio', 'lado', 'lago', 'lamento', 'lampara',
        'lanza', 'lapiz', 'lapso', 'largo', 'larva', 'laser', 'lastima',
        'lateral', 'latitud', 'laurel', 'lava', 'leal', 'leccion', 'lector',
        'legado', 'legal', 'leyenda', 'liberal', 'libertad', 'libro',
        'licencia', 'licor', 'lider', 'lienzo', 'liga', 'limite', 'limon',
        'linea', 'lino', 'linterna', 'liquido', 'lista', 'literal', 'litro',
        'llave', 'lobo', 'local', 'logica', 'logotipo', 'lona', 'longitud',
        'lote', 'loto', 'lucro', 'lugar', 'lujo', 'luna', 'lupa', 'luto', 'luz',
        'macho', 'madera', 'madre', 'maestro', 'mafia', 'magia', 'magma',
        'magneto', 'mago', 'maiz', 'malla', 'mama', 'mamifero', 'manager',
        'mancha', 'mandato', 'mango', 'mania', 'mano', 'mansion', 'manto',
        'manual', 'mapa', 'maqueta', 'maquina', 'mar', 'marca', 'marco',
        'marea', 'marfil', 'margen', 'marina', 'mariposa', 'martillo', 'martir',
        'masa', 'masaje', 'mascara', 'mascota', 'mastil', 'materia', 'matriz',
        'maximo', 'mayor', 'medalla', 'media', 'medico', 'medio', 'medula',
        'melodia', 'melon', 'memoria', 'mencion', 'mensaje', 'menu', 'mercado',
        'merito', 'mes', 'mesa', 'meta', 'metal', 'metodo', 'metro', 'mezcla',
        'miedo', 'miel', 'miembro', 'milagro', 'milla', 'mina', 'mineral',
        'minimo', 'ministro', 'minuto', 'mirada', 'misa', 'misil', 'mision',
        'misterio', 'mitad', 'mito', 'mixto', 'moda', 'modelo', 'modulo',
        'moho', 'molde', 'molecula', 'momento', 'momia', 'moneda', 'monitor',
        'mono', 'monstruo', 'montana', 'monte', 'moral', 'mordisco', 'morral',
        'mortal', 'mosaico', 'mosquito', 'mostrador', 'motel', 'motin', 'motor',
        'mousse', 'movil', 'muchedumbre', 'mueble', 'muerte', 'muestra',
        'mujer', 'mula', 'multa', 'mundo', 'municion', 'muro', 'musculo',
        'museo', 'musgo', 'musica', 'muslo', 'mutacion', 'mutuo', 'nacion',
        'nada', 'nariz', 'natal', 'nativo', 'naufragio', 'navaja', 'naval',
        'nave', 'navidad', 'necesidad', 'negocio', 'neon', 'nervio', 'nexo',
        'nicho', 'nido', 'niebla', 'nieto', 'nieve', 'ninfa', 'nino', 'nivel',
        'no', 'noble', 'noche', 'nodo', 'nodriza', 'nomada', 'nombre', 'nomina',
        'noria', 'norma', 'norte', 'nota', 'noticia', 'novela', 'novia',
        'novio', 'nube', 'nucleo', 'nudo', 'nuera', 'numero', 'oasis',
        'obelisco', 'obispo', 'objeto', 'oblea', 'obra', 'obrero', 'ocasion',
        'oceano', 'octava', 'oculto', 'ocupacion', 'ocurrencia', 'oda', 'odio',
        'oeste', 'ofensa', 'oferta', 'oficial', 'oficina', 'oficio', 'ogro',
        'oido', 'ojo', 'ola', 'oleada', 'oleo', 'olfato', 'oliva', 'olivo',
        'olla', 'olmo', 'olor', 'olvido', 'ombligo', 'omega', 'omision',
        'omnibus', 'onza', 'opaco', 'opcion', 'opera', 'operacion', 'opinion',
        'opio', 'oponente', 'optica', 'opuesto', 'oracion', 'oraculo', 'orador',
        'oral', 'orbita', 'orden', 'oreja', 'orfebre', 'organo', 'orgasmo',
        'orgia', 'orgullo', 'oriente', 'origen', 'orilla', 'orina', 'orla',
        'oro', 'orquesta', 'ortiga', 'oruga', 'orzuelo', 'oso', 'ostra',
        'otono', 'ovalo', 'ovario', 'oveja', 'ovni', 'ovulo', 'oxido',
        'oxigeno', 'oyente', 'ozono'},
 'fr': {'acolyte', 'ah', 'ambivalence', 'art', 'ballet', 'bar', 'base', 'bazar',
        'blouse', 'bon', 'border', 'bravo', 'bureau', 'bus', 'cable', 'cafe',
        'camp', 'canal', 'cancer', 'canon', 'caoutchouc', 'capital', 'carte',
        'centre', 'champion', 'chance', 'chaos', 'chef', 'choc', 'cinema',
        'cirque', 'client', 'climat', 'climax', 'club', 'code', 'college',
        'colonel', 'combat', 'comedie', 'comite', 'compas', 'concert',
        'contact', 'continent', 'contraire', 'contrat', 'controle', 'copie',
        'cordon', 'corps', 'correct', 'costume', 'coton', 'coude', 'couleur',
        'couloir', 'coup', 'couple', 'cour', 'courage', 'court', 'cousin',
        'couteau', 'couvercle', 'crabe', 'cravate', 'crayon', 'creme', 'crepe',
        'creux', 'cri', 'crime', 'crise', 'crochet', 'croix', 'croute', 'cube',
        'cuir', 'cuisine', 'cuisse', 'cuivre', 'culotte', 'culte', 'culture',
        'cure', 'cuve', 'cygne', 'dame', 'danger', 'danse', 'date', 'dauphin',
        'debat', 'debut', 'deces', 'dechet', 'decision', 'decor', 'decret',
        'dedain', 'defaite', 'defaut', 'defense', 'defi', 'deficit', 'degat',
        'degre', 'delai', 'delice', 'delit', 'deluge', 'demain', 'demande',
        'demence', 'demeure', 'demi', 'democratie', 'demon', 'dent', 'depart',
        'depense', 'depot', 'deroule', 'desert', 'desir', 'desordre', 'dessein',
        'dessert', 'dessin', 'destin', 'detail', 'detective', 'detente',
        'detour', 'dette', 'deuil', 'devis', 'devoir', 'diable', 'dialogue',
        'diamant', 'dieu', 'diffuseur', 'digue', 'dilemme', 'dimanche',
        'dindon', 'diner', 'diplome', 'directeur', 'direction', 'disciple',
        'discipline', 'discours', 'disque', 'distance', 'district', 'divan',
        'docteur', 'doctrine', 'document', 'doigt', 'domaine', 'domicile',
        'domino', 'dommage', 'don', 'donjon', 'dortoir', 'doryphore', 'dose',
        'dossier', 'dot', 'douane', 'double', 'doute', 'dragon', 'drap',
        'drapeau', 'drogue', 'droit', 'dromadaire', 'duc', 'duel', 'dune',
        'duo', 'duvet', 'eau', 'ebene', 'eboulement', 'ecaille', 'ecart',
        'echange', 'echantillon', 'echarpe', 'echec', 'echelle', 'echelon',
        'echo', 'eclair', 'eclat', 'eclipse', 'eclisse', 'ecluse', 'ecole',
        'ecorce', 'ecran', 'ecrit', 'ecrou', 'ecume', 'ecurie', 'editeur',
        'edition', 'effet', 'effort', 'egard', 'eglise', 'egout', 'elan',
        'element', 'elephant', 'eleve', 'elite', 'eloge', 'email', 'emballage',
        'embarras', 'embauche', 'embouteillage', 'embuscade', 'emeute',
        'emission', 'emotion', 'empechement', 'empereur', 'empire', 'emploi',
        'empreinte', 'emprunt', 'encadrement', 'enceinte', 'encensoir',
        'enclave', 'enclos', 'enclume', 'encre', 'enfant', 'enfer', 'engin',
        'enigme', 'enjeu', 'ennemi', 'ennui', 'enquete', 'ensemble', 'entente',
        'enthousiasme', 'entonnoir', 'entrain', 'entrave', 'entree',
        'entreprise', 'entretien', 'enveloppe', 'envergure', 'envie',
        'epaisseur', 'epaule', 'epee', 'epervier', 'epi', 'epice', 'epilogue',
        'epingle', 'episode', 'eponge', 'epoque', 'epouse', 'epreuve',
        'eprouvette', 'equilibre', 'equipe', 'equipement', 'equite',
        'equivalent', 'ere', 'ermite', 'erreur', 'escalier', 'escargot',
        'esclave', 'escompte', 'escrime', 'espace', 'espece', 'esperance',
        'espion', 'espoir', 'esprit', 'esquisse', 'essai', 'essence', 'essieu',
        'essor', 'estimable', 'estomac', 'estrade', 'etape', 'etat', 'etau',
        'ete', 'eteignoir', 'etendard', 'etiquette', 'etoffe', 'etoile',
        'etonnement', 'etouffement', 'etrier', 'etroit', 'etude', 'etudiant',
        'etui', 'etymologie', 'eucalyptus', 'evenement', 'eventail', 'eviction',
        'evidence', 'evolution', 'exactitude', 'examen', 'exces', 'excitation',
        'exclamation', 'excursion', 'excuse', 'execution', 'exemplaire',
        'exemple', 'exercice', 'exil', 'existence', 'exode', 'expansion',
        'experience', 'expert', 'explication', 'exploit', 'exploration',
        'explosion', 'exportation', 'expose', 'expression', 'extase',
        'extension', 'exterieur', 'extinction', 'extrait', 'extreme',
        'extremite', 'fable', 'fabricant', 'fabrication', 'facade', 'face',
        'facette', 'facteur', 'facture', 'faculte', 'fagot', 'faiblesse',
        'faim', 'faisceau', 'famille', 'fanatique', 'fanfare', 'fantasme',
        'faon', 'fardeau', 'farine', 'faste', 'fatalite', 'fatigue', 'faubourg',
        'faucon', 'faute', 'fauteuil', 'fauve', 'faveur', 'favori', 'fecondite',
        'federation', 'fee', 'feinte', 'felicitation', 'femelle', 'femme',
        'fenetre', 'fente', 'fer', 'ferme', 'fermeture', 'ferveur', 'fete',
        'feu', 'feuille', 'feutre', 'feve', 'fiance', 'fibre', 'ficelle',
        'fiche', 'fidelite', 'fief', 'fierte', 'fievre', 'figure', 'fil',
        'filament', 'filet', 'fille', 'film', 'fils', 'filtre', 'fin',
        'finance', 'finesse', 'fiole', 'firme', 'flacon', 'flamme', 'flanc',
        'flaque', 'flash', 'fleau', 'fleche', 'fleur', 'fleuve', 'flexibilite',
        'floc', 'flocon', 'flot', 'flotte', 'fluide', 'flux', 'foi', 'foie',
        'foire', 'folie', 'fonction', 'fond', 'fondation', 'fonds', 'fontaine',
        'fonte', 'football', 'force', 'foret', 'forfait', 'forge', 'forme',
        'formule', 'fort', 'forteresse', 'fortune', 'forum', 'fosse', 'fossile',
        'fou', 'foudre', 'fouille', 'foule', 'four', 'fourche', 'fourgon',
        'fourmi', 'fourneau', 'fournisseur', 'fourrure', 'foyer', 'fracas',
        'fraction', 'fracture', 'fragilite', 'fragment', 'fraicheur', 'fraise',
        'franchise', 'frange', 'fraternite', 'fraude', 'frayeur', 'frein',
        'frequence', 'frere', 'fresque', 'friandise', 'friction', 'frigo',
        'frisson', 'froid', 'froment', 'front', 'frontiere', 'frottement',
        'fruit', 'fuite', 'fumee', 'fumier', 'fureur', 'furie', 'fusil',
        'fusion', 'fustigation', 'fut', 'futur', 'goutte', 'me', 'oh', 'ok'},
 'nl': {'abstract', 'absurd', 'ambivalent', 'april', 'arrogant', 'astronaut',
        'baby', 'bal', 'ballet', 'band', 'bank', 'bar', 'basis', 'bed', 'bier',
        'blouse', 'boot', 'bos', 'boter', 'brief', 'broek', 'brood', 'brug',
        'buik', 'bureau', 'bus', 'cake', 'camera', 'camping', 'cannabis',
        'chaos', 'chef', 'chocolade', 'cinema', 'circus', 'club', 'code',
        'coherent', 'college', 'comfort', 'compact', 'competent', 'complex',
        'compliant', 'component', 'computer', 'concept', 'concert', 'conflict',
        'congruent', 'consensus', 'consistent', 'constant', 'consul', 'contact',
        'container', 'context', 'contingent', 'contract', 'convenant',
        'corridor', 'crisis', 'cultuur', 'curriculum', 'cursus', 'dank', 'dans',
        'datum', 'decibel', 'defect', 'depot', 'deur', 'diamant', 'diaspora',
        'dier', 'dilemma', 'diner', 'diploma', 'directeur', 'directie',
        'discotheek', 'discussie', 'dissident', 'distributie', 'district',
        'doctor', 'doel', 'dokter', 'domein', 'dorp', 'dosis', 'dossier',
        'drama', 'drank', 'drift', 'droogte', 'droom', 'drug', 'duel',
        'dumping', 'duur', 'dwerg', 'dynamiek', 'echo', 'economie', 'edict',
        'effect', 'eiland', 'einde', 'element', 'elite', 'embargo', 'emotie',
        'energie', 'entourage', 'envelop', 'episode', 'erf', 'erfenis',
        'escalatie', 'escort', 'escorte', 'essay', 'essentie', 'etiket',
        'examen', 'excerpt', 'experiment', 'expert', 'expertise', 'explosie',
        'export', 'extreem', 'fabriek', 'facsimile', 'factor', 'familie', 'fan',
        'fantasie', 'fase', 'fauna', 'feedback', 'feest', 'feit', 'festival',
        'film', 'filter', 'finale', 'financier', 'flacon', 'flat', 'fles',
        'flits', 'flora', 'fluit', 'flux', 'focus', 'folder', 'folklore',
        'fonds', 'formaat', 'formule', 'fornuis', 'fort', 'foto', 'fout',
        'fractie', 'fragment', 'frame', 'frequentie', 'fruit', 'functie',
        'fusie', 'gadget', 'gang', 'garage', 'garderobe', 'gas', 'gast',
        'gebaar', 'gebied', 'gebit', 'geboorte', 'gedachte', 'gedrag', 'geest',
        'geit', 'geld', 'geloof', 'geluid', 'geluk', 'gemak', 'genesis',
        'genie', 'genocide', 'genre', 'geschenk', 'geschil', 'gesprek', 'geur',
        'gevaar', 'gevoel', 'geweld', 'gewicht', 'gewoonte', 'gezag', 'gezicht',
        'gezin', 'gids', 'gif', 'gitaar', 'glas', 'glorie', 'golf', 'gordijn',
        'goud', 'graad', 'graf', 'grafiek', 'gram', 'grandeur', 'gras', 'grens',
        'griep', 'grijns', 'groep', 'grond', 'grot', 'guillotine', 'gunst',
        'haak', 'haar', 'haard', 'haast', 'habitat', 'hagel', 'hal', 'half',
        'hals', 'ham', 'hamer', 'hand', 'handel', 'handschoen', 'hard',
        'haring', 'hart', 'haven', 'haver', 'hectare', 'heerser', 'heffing',
        'heft', 'heg', 'heid', 'hek', 'held', 'helikopter', 'helm', 'hemd',
        'hemel', 'hennep', 'herfst', 'herinnering', 'hert', 'heuvel', 'hiaat',
        'hiel', 'hijskraan', 'hinder', 'hond', 'honing', 'hoofd', 'hoogte',
        'hooi', 'hoop', 'hoorn', 'horizon', 'horloge', 'hospitaal', 'hotel',
        'hout', 'huid', 'huig', 'huis', 'humor', 'huur', 'huwelijk', 'hype',
        'hypotheek', 'hypothese', 'icoon', 'idee', 'idioom', 'ijs', 'ijzer',
        'illusie', 'imago', 'imam', 'impact', 'import', 'impuls', 'incident',
        'indicatie', 'individu', 'industrie', 'infectie', 'inflatie', 'ingang',
        'inhoud', 'injectie', 'inkomen', 'inkt', 'inloop', 'inrichting',
        'insect', 'instinct', 'instrument', 'inval', 'invloed', 'ironie',
        'item', 'ivoor', 'ladder', 'lamp', 'laptop', 'latent', 'lateral',
        'letter', 'link', 'machine', 'mainstream', 'maker', 'manifold', 'meter',
        'mild', 'modern', 'monitor', 'navel', 'nest', 'normal', 'pan', 'park',
        'partner', 'pen', 'permanent', 'piano', 'plan', 'printer', 'project',
        'ring', 'school', 'shirt', 'sober', 'station', 'storm', 'student',
        'talent', 'tent', 'tram', 'turbulent', 'uniform', 'urgent', 'variant',
        'water', 'week', 'weekend', 'wind', 'winter'},
 'sv': {'advent', 'all', 'altare', 'ambition', 'ammunition', 'ankare',
        'apelsin', 'apostel', 'april', 'arena', 'argument', 'argumentation',
        'arkiv', 'arm', 'aroma', 'artikel', 'artist', 'asfalt', 'ask',
        'aspirant', 'astma', 'astronaut', 'atlas', 'attack', 'attityd',
        'avfall', 'avsikt', 'axel', 'back', 'backen', 'bacon', 'bad', 'bagage',
        'bageri', 'balans', 'balett', 'balkong', 'balsam', 'balte', 'banan',
        'band', 'bandy', 'bank', 'bar', 'barack', 'baron', 'barr', 'barriar',
        'bas', 'basar', 'basta', 'bastu', 'bat', 'bataljon', 'batteri', 'bebis',
        'bensin', 'berg', 'bestick', 'betong', 'betyg', 'bibel', 'bibliotek',
        'biff', 'bil', 'bild', 'biljard', 'biljett', 'biologi', 'biskop', 'bit',
        'bjorn', 'blad', 'blankett', 'blixt', 'block', 'blockad', 'blomma',
        'blondin', 'blus', 'bly', 'bo', 'bofink', 'boggie', 'boj', 'bok',
        'boka', 'bolag', 'boll', 'bomb', 'bonde', 'borg', 'borste', 'bostad',
        'botten', 'box', 'brand', 'bransch', 'brant', 'bredd', 'brev', 'bricka',
        'bro', 'bror', 'brott', 'brud', 'bruk', 'brunn', 'brygga', 'budget',
        'bukt', 'bulle', 'bunt', 'buss', 'butik', 'by', 'bygd', 'bygdegard',
        'bygel', 'byggnad', 'byxa', 'cafe', 'camping', 'cancer', 'cell',
        'cement', 'central', 'centrum', 'ceremoni', 'champagne', 'chans',
        'chaos', 'chassi', 'check', 'chef', 'chili', 'chips', 'chock',
        'choklad', 'cigarett', 'cirkel', 'cirkus', 'citat', 'citrus',
        'civilisation', 'clown', 'cocktail', 'cola', 'container', 'cool',
        'cykel', 'cylinder', 'cymbal', 'dack', 'dag', 'dalj', 'damm', 'dans',
        'darr', 'data', 'dator', 'datum', 'debatt', 'debut', 'december',
        'decimal', 'definition', 'deg', 'dekad', 'deklaration', 'del',
        'delegation', 'demokrati', 'demon', 'demonstration', 'depa',
        'departement', 'design', 'dessert', 'detalj', 'detektiv', 'dialog',
        'diamant', 'diet', 'dike', 'dilemma', 'dimension', 'dimma', 'diplom',
        'diploma', 'diplomat', 'direktion', 'direktiv', 'direktor', 'disciplin',
        'disk', 'diskussion', 'distans', 'distrikt', 'dito', 'divan',
        'division', 'dock', 'docka', 'doktor', 'dokument', 'dold', 'dolk',
        'dollar', 'dom', 'domare', 'domkyrka', 'domstol', 'don', 'donation',
        'dopp', 'dorr', 'dos', 'dosa', 'dotter', 'dovhet', 'dragg', 'drake',
        'drama', 'drastisk', 'drev', 'drift', 'drom', 'droppe', 'drottning',
        'dryck', 'dubb', 'duell', 'duett', 'duk', 'dunge', 'duns', 'dusch',
        'duva', 'dvarg', 'dygn', 'dyna', 'dynamik', 'dynastin', 'dynga',
        'dyrkan', 'ebba', 'effekt', 'eftermiddag', 'efternamn', 'eftertanke',
        'egendom', 'egenhet', 'ek', 'ekonomi', 'ekorre', 'eld', 'elefant',
        'elegans', 'element', 'elev', 'elit', 'emotion', 'energi', 'england',
        'enhet', 'enkelhet', 'enkrona', 'entre', 'entusiasm', 'epidemi', 'epok',
        'era', 'erfarenhet', 'erik', 'erotik', 'erovring', 'ert', 'eskader',
        'eskort', 'espresso', 'ess', 'essa', 'estet', 'etapp', 'etik',
        'etikett', 'etta', 'ettring', 'eunuck', 'europa', 'evangelium',
        'evolution', 'examen', 'examination', 'exempel', 'exemplar', 'exil',
        'existens', 'expansion', 'experiment', 'expert', 'explosion', 'export',
        'express', 'extra', 'extrem', 'fabrik', 'fack', 'fackla', 'fader',
        'fagg', 'faktor', 'faktum', 'faktura', 'falk', 'fall', 'falla', 'falle',
        'falskhet', 'familj', 'fan', 'fanfar', 'fantasie', 'fara', 'farg',
        'farkost', 'farm', 'fart', 'fartyg', 'fas', 'fasa', 'fasan', 'fason',
        'faste', 'fastighet', 'fatta', 'fauna', 'favorit', 'fe', 'feber',
        'februari', 'federation', 'feghet', 'fel', 'fela', 'fem', 'femma',
        'femtiolapp', 'fena', 'fenomen', 'fest', 'festival', 'fett', 'fiber',
        'ficka', 'figur', 'fika', 'fikus', 'fil', 'film', 'filosofi', 'filter',
        'filur', 'finans', 'finess', 'finger', 'fink', 'fira', 'fisk', 'fiske',
        'fjader', 'fjall', 'fjard', 'fjart', 'fjol', 'fjor', 'fjorton', 'flack',
        'flagg', 'flagga', 'flak', 'flakt', 'flamma', 'flaska', 'flata',
        'fleece', 'flera', 'flik', 'flinga', 'flint', 'flock', 'flod', 'flora',
        'fluga', 'flyg', 'flygel', 'flykt', 'fodelse', 'foder', 'fogde',
        'fokus', 'folje', 'folk', 'fond', 'fonster', 'fordon', 'forell',
        'forlust', 'form', 'formaga', 'formalitet', 'format', 'formel',
        'fornuft', 'forr', 'forrad', 'fors', 'forskning', 'forsvar', 'fort',
        'fortret', 'forum', 'fossil', 'foster', 'fot', 'foto', 'frack', 'fraga',
        'fragment', 'frakt', 'framgang', 'framtid', 'fran', 'fras', 'fred',
        'frekvens', 'fresk', 'fria', 'frid', 'frilla', 'frisyr', 'fritid',
        'frivillig', 'fro', 'frokost', 'frossa', 'frost', 'frukt', 'fruntimmer',
        'fuga', 'fukt', 'fullmakt', 'fundament', 'funktion', 'fura', 'furu',
        'fusion', 'fusk', 'fysik', 'gangster', 'garage', 'gas', 'hand', 'hare',
        'ideal', 'idiot', 'in', 'information', 'institution', 'integration',
        'journalist', 'juice', 'ketchup', 'lunch', 'media', 'merit',
        'motivation', 'museum', 'norm', 'observation', 'odds', 'opposition',
        'panel', 'partner', 'passion', 'pasta', 'pension', 'person', 'position',
        'press', 'problem', 'process', 'region', 'religion', 'reporter', 'risk',
        'salt', 'standard', 'stockholm', 'student', 'symposium', 'system',
        'temperament', 'term', 'test', 'text', 'tiger', 'tradition',
        'transport', 'under', 'vision', 'workshop'}}

ENGLISH_FUNCTION_PREFIXES = (
    "to ",
    "the ",
    "a ",
    "an ",
    "of ",
    "in ",
    "at ",
    "on ",
    "by ",
    "for ",
    "with ",
    "from ",
    "his ",
    "her ",
    "their ",
    "our ",
    "my ",
    "your ",
    "its ",
)

TSV_HEADER = [
    "lemma",
    "pos",
    "english_gloss",
    "english_pos",
    "cefr",
    "rank",
    "concept_key",
]


@dataclass(frozen=True)
class Row:
    lang: str
    lemma: str
    pos: str
    english_gloss: str
    rank: int | None


def normalize_gloss(gloss: str) -> str:
    """Match the app's key: lower, drop a leading 'to ', strip spaces only."""
    return re.sub("^to ", "", gloss.lower()).strip(" ")


def _unquote(value: str) -> str:
    value = value.strip()
    if value.startswith('"') and value.endswith('"'):
        value = value[1:-1].replace('""', '"')
    return value


def load_csv_rows(root: Path) -> list[Row]:
    """Read the authored CEFR CSVs (levels A1-C1) of every language."""
    rows: list[Row] = []
    for name, code in LANG_DIRS.items():
        for level in LEVELS:
            path = root / name / f"{level}.csv"
            if not path.exists():
                continue
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.reader(handle)
                next(reader, None)
                for cols in reader:
                    lemma = _unquote(cols[0]) if len(cols) > 0 else ""
                    if not lemma:
                        continue
                    gloss = _unquote(cols[1]) if len(cols) > 1 else ""
                    pos = cols[3].strip() if len(cols) > 3 else ""
                    rows.append(Row(code, lemma, pos, gloss, None))
        path = root / name / "expansion.csv"
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            for cols in reader:
                lemma = _unquote(cols[0]) if len(cols) > 0 else ""
                if not lemma:
                    continue
                gloss = _unquote(cols[1]) if len(cols) > 1 else ""
                pos = cols[3].strip() if len(cols) > 3 else ""
                rows.append(Row(code, lemma, pos, gloss, None))
    return rows


def load_delivery_rows(delivery: Path) -> list[Row]:
    """Read contract TSVs; the filename is the language (dir name or code)."""
    rows: list[Row] = []
    for path in sorted(delivery.glob("*.tsv")):
        code = LANG_DIRS.get(path.stem, path.stem)
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle, delimiter="\t")
            header = next(reader, None)
            if header != TSV_HEADER:
                raise ValueError(f"{path.name}: unexpected header {header}")
            for cols in reader:
                if not cols or not cols[0].strip():
                    continue
                if len(cols) < 4:
                    raise ValueError(
                        f"{path.name}: row has {len(cols)} columns: {cols}"
                    )
                rank_text = cols[5].strip() if len(cols) > 5 else ""
                rows.append(
                    Row(
                        lang=code,
                        lemma=cols[0].strip(),
                        pos=cols[3].strip(),
                        english_gloss=cols[2].strip(),
                        rank=int(rank_text) if rank_text else None,
                    )
                )
    return rows


def check_shrunken_glosses(delivery: list[Row], source: list[Row]) -> list[str]:
    """Criterion 1: a multi-word source gloss must not shrink to one word.

    A lemma may carry several senses in the source; every source gloss is
    kept. A single-word delivery gloss is a shrink only when the source has a
    multi-word gloss for that lemma and the single word is not itself one of
    the authored senses.
    """
    source_glosses: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in source:
        source_glosses[(row.lang, row.lemma)].add(row.english_gloss)
    violations = []
    for row in delivery:
        originals = source_glosses.get((row.lang, row.lemma), set())
        multi = [g for g in originals if len(g.split()) > 1]
        if not multi:
            continue
        if len(row.english_gloss.split()) == 1 and row.english_gloss not in originals:
            violations.append(
                f"{row.lang}:{row.lemma}: {sorted(originals)} -> '{row.english_gloss}'"
            )
    return violations


def check_duplicates(rows: list[Row]) -> list[str]:
    """Criterion 2: (lang, lemma, gloss_norm) is unique."""
    counts = Counter((r.lang, r.lemma, normalize_gloss(r.english_gloss)) for r in rows)
    return [f"{k[0]}:{k[1]}:{k[2]}" for k, n in counts.items() if n > 1]


def check_rank_gaps(rows: list[Row]) -> list[str]:
    """Criterion 3: exactly one rank-1 row per (lang, gloss_norm, pos)."""
    groups: dict[tuple[str, str, str], int] = defaultdict(int)
    for row in rows:
        if row.rank == 1:
            groups[(row.lang, normalize_gloss(row.english_gloss), row.pos)] += 1
    seen = {(r.lang, normalize_gloss(r.english_gloss), r.pos) for r in rows}
    violations = []
    for key in sorted(seen):
        if groups.get(key, 0) != 1:
            violations.append(f"{key[0]}:{key[1]}:{key[2]}")
    return violations


def pair_coverage(rows: list[Row], source: str, target: str) -> tuple[int, int, int]:
    """Criterion 4 per pair: (eindeutig, mehrdeutig, ohne)."""
    index: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        if row.lang == target:
            index[(normalize_gloss(row.english_gloss), row.pos)].add(row.lemma)
    eindeutig = mehrdeutig = ohne = 0
    for row in rows:
        if row.lang != source:
            continue
        gloss = normalize_gloss(row.english_gloss)
        if not gloss:
            ohne += 1
            continue
        found = len(index.get((gloss, row.pos), set()))
        if found == 1:
            eindeutig += 1
        elif found > 1:
            mehrdeutig += 1
        else:
            ohne += 1
    return eindeutig, mehrdeutig, ohne


def check_ascii(rows: list[Row]) -> list[str]:
    """Criterion 5: the pivot gloss is English ASCII (or blank)."""
    violations = []
    for row in rows:
        gloss = row.english_gloss
        if not gloss or not GLOSS_ASCII.match(gloss):
            violations.append(f"{row.lang}:{row.lemma}: '{gloss}'")
    return violations


def check_script_and_substance(rows: list[Row]) -> list[str]:
    """Criterion 6: valid target script, no junk placeholder tokens, no ungrounded gloss copies."""
    violations = []
    for row in rows:
        lemma = row.lemma.strip()
        if not lemma:
            violations.append(f"{row.lang}:empty_lemma:'{row.english_gloss}'")
            continue
        if lemma in FORBIDDEN_JUNK_LEMMAS:
            violations.append(f"{row.lang}:junk_lemma:'{lemma}'")
            continue
        if row.lang == "ar":
            if any(c.isalpha() for c in lemma) and not ARABIC_SCRIPT.search(lemma):
                violations.append(f"ar:non_arabic_script:'{lemma}'")
        elif row.lang == "zh":
            if any(c.isalpha() for c in lemma) and not CHINESE_SCRIPT.search(lemma):
                violations.append(f"zh:non_chinese_script:'{lemma}'")
        elif row.lang != "en":
            # Non-English Latin-script languages: check for ungrounded English gloss copies
            gloss_norm = normalize_gloss(row.english_gloss)
            lemma_norm = normalize_gloss(lemma)
            lemma_lower = lemma.lower()
            gloss_lower = row.english_gloss.lower()

            # Check if lemma copies the English gloss
            is_gloss_copy = (lemma_norm == gloss_norm) or (lemma_lower == gloss_lower)
            if is_gloss_copy:
                allowlist = COGNATE_ALLOWLIST.get(row.lang, set())
                if lemma_lower not in allowlist and lemma_norm not in allowlist:
                    violations.append(f"{row.lang}:ungrounded_gloss_copy:'{lemma}'")
                    continue

            # 2. English function word prefix copies (e.g. lemma starts with "to ", "the ", "his ")
            if lemma_lower.startswith(ENGLISH_FUNCTION_PREFIXES):
                violations.append(f"{row.lang}:english_function_prefix_copy:'{lemma}'")
    return violations


def run_baseline(root: Path) -> int:
    rows = load_csv_rows(root)
    print(f"CSV baseline: {len(rows)} rows")
    print("criterion 1: n/a (needs a delivery)")
    dups = check_duplicates(rows)
    print(f"criterion 2: {len(dups)} duplicate keys")
    print("criterion 3: n/a (CSVs carry no rank)")
    print("criterion 4: eindeutige Abdeckung je Sprachpaar")
    codes = sorted(set(LANG_DIRS.values()))
    for source in codes:
        for target in codes:
            if source == target:
                continue
            ein, mehr, ohne = pair_coverage(rows, source, target)
            total = ein + mehr + ohne
            ratio = ein / total if total else 0.0
            print(
                f"  {source}->{target}: {ein}/{total} = {ratio:.0%} "
                f"(mehrdeutig {mehr}, ohne {ohne})"
            )
    ascii_bad = check_ascii(rows)
    print(f"criterion 5: {len(ascii_bad)} non-ASCII or blank glosses")
    return 0


def run_delivery(delivery: Path, root: Path) -> int:
    rows = load_delivery_rows(delivery)
    if not rows:
        print(f"no TSV rows found in {delivery}")
        return 1
    source = load_csv_rows(root)
    failed = False

    shrunk = check_shrunken_glosses(rows, source)
    print(f"criterion 1: {len(shrunk)} shrunken multi-word glosses")
    for item in shrunk[:20]:
        print(f"  {item}")
    failed |= bool(shrunk)

    dups = check_duplicates(rows)
    print(f"criterion 2: {len(dups)} duplicate keys")
    for item in dups[:20]:
        print(f"  {item}")
    failed |= bool(dups)

    gaps = check_rank_gaps(rows)
    print(f"criterion 3: {len(gaps)} group(s) without exactly one rank 1")
    for item in gaps[:20]:
        print(f"  {item}")
    failed |= bool(gaps)

    print("criterion 4: eindeutige Abdeckung je Sprachpaar (>= 60 %)")
    codes = sorted({r.lang for r in rows})
    for source_lang in codes:
        for target in codes:
            if source_lang == target:
                continue
            ein, mehr, ohne = pair_coverage(rows, source_lang, target)
            total = ein + mehr + ohne
            ratio = ein / total if total else 0.0
            status = "OK " if ratio >= MIN_COVERAGE else "LOW"
            print(
                f"  [{status}] {source_lang}->{target}: {ein}/{total} "
                f"= {ratio:.0%} (mehrdeutig {mehr}, ohne {ohne})"
            )
            failed |= ratio < MIN_COVERAGE

    ascii_bad = check_ascii(rows)
    print(f"criterion 5: {len(ascii_bad)} non-ASCII or blank glosses")
    for item in ascii_bad[:20]:
        print(f"  {item}")
    failed |= bool(ascii_bad)

    script_bad = check_script_and_substance(rows)
    print(f"criterion 6: {len(script_bad)} script or substance violations")
    for item in script_bad[:20]:
        print(f"  {item}")
    failed |= bool(script_bad)

    print("RESULT: FAIL" if failed else "RESULT: PASS")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "delivery",
        nargs="?",
        help="directory with one contract TSV per language",
    )
    args = parser.parse_args(argv)
    if args.delivery:
        return run_delivery(Path(args.delivery), ROOT)
    return run_baseline(ROOT)


if __name__ == "__main__":
    sys.exit(main())
