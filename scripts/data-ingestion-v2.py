import os
import glob
import html
import io
import re
from PyPDF2 import PdfReader, PdfWriter
from azure.identity import AzureDeveloperCliCredential
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import *
from azure.search.documents import SearchClient
from azure.ai.formrecognizer import DocumentAnalysisClient
from dotenv import load_dotenv

MAX_SECTION_LENGTH = 1000
SENTENCE_SEARCH_LIMIT = 100
SECTION_OVERLAP = 100


class AzureCredentials:
    def __init__(self, blob, form, search, search_index, container):
        self.storageaccount = blob
        self.formrecognizerservice = form
        self.searchservice = search
        self.index = search_index
        self.container = container
        self.category = None


load_dotenv("C:/Users/agustmio/Desktop/GALICIA/.azure/cd/.env")

args = AzureCredentials(
    os.getenv("AZURE_STORAGE_ACCOUNT"),
    os.getenv("AZURE_FORMRECOGNIZER_SERVICE"),
    os.getenv("AZURE_SEARCH_SERVICE"),
    os.getenv("AZURE_SEARCH_INDEX"),
    os.getenv("AZURE_STORAGE_CONTAINER"),
)
print(args.searchservice)
print(args.storageaccount)
print(args.formrecognizerservice)
DATA_PATH = "C:/Users/agustmio/Desktop/GALICIA/data/*"

FORM_KEY = os.getenv("AZURE_FORMRECOGNIZER_KEY")
SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
creds = AzureDeveloperCliCredential()
search_creds = AzureKeyCredential(SEARCH_KEY)  # AzureDeveloperCliCredential()
formrecognizer_creds = AzureKeyCredential(FORM_KEY)  # AzureDeveloperCliCredential()
storage_creds = os.getenv("AZURE_STORAGE_CREDENTIAL")  # AzureDeveloperCliCredential()


def table_to_html(table):
    table_html = "<table>"
    rows = [
        sorted(
            [cell for cell in table.cells if cell.row_index == i],
            key=lambda cell: cell.column_index,
        )
        for i in range(table.row_count)
    ]
    for row_cells in rows:
        table_html += "<tr>"
        for cell in row_cells:
            tag = (
                "th"
                if (cell.kind == "columnHeader" or cell.kind == "rowHeader")
                else "td"
            )
            cell_spans = ""
            if cell.column_span > 1:
                cell_spans += f" colSpan={cell.column_span}"
            if cell.row_span > 1:
                cell_spans += f" rowSpan={cell.row_span}"
            table_html += f"<{tag}{cell_spans}>{html.escape(cell.content)}</{tag}>"
        table_html += "</tr>"
    table_html += "</table>"
    return table_html


def extract_dni(text):
    dni_pattern = r"D\.N\.I\.\s*(\d{8})"
    dni_match = re.search(dni_pattern, text)
    if dni_match:
        dni_value = dni_match.group(1)  # Se almacena el valor del DNI en una variable
        return dni_value
    else:
        dni_pattern = r"PRUER\s*(\d{8})"  # Hay algunos dni que estan despues de esas siglas, es un tema del form recognizer
        dni_match = re.search(dni_pattern, text)
        if dni_match:
            dni_value = dni_match.group(
                1
            )  # Se almacena el valor del DNI en una variable
            return dni_value
        else:
            return None


def extract_cuit(text):
    cuit_pattern = r"C\.U\.I\.T\.\s*(\d{2}-\d{8}-\d)"
    cuit_match = re.search(cuit_pattern, text)
    if cuit_match:
        cuit_value = cuit_match.group(1)  # Almacena el valor del CUIT en una variable
        return cuit_value
    else:
        cuit_pattern = r"C\.U\.I\.T\. PRUER\s*(\d{2}-\d{8}-\d)"
        cuit_match = re.search(cuit_pattern, text)
        if cuit_match:
            cuit_value = cuit_match.group(1)
            return cuit_value
        else:
            return None


def extract_npoliza(text):
    npoliza_pattern = r"(?:PÓLIZA(?:\sN°)?\s)?(\d{3}-\d{8}-\d{2})"
    npoliza_match = re.search(npoliza_pattern, text, re.IGNORECASE)
    if npoliza_match:
        npoliza_value = npoliza_match.group(1)
        return npoliza_value
    else:
        npoliza_pattern = r"PÓLIZA\sN°\s(\d{3}-\d{8}-\d{2})"
        npoliza_match = re.search(npoliza_pattern, text, re.IGNORECASE)
        if npoliza_match:
            npoliza_value = npoliza_match.group(
                1
            )  # Almacena el valor de npoliza en una variable
            return npoliza_value
        else:
            return None


def get_document_text(filename):
    offset = 0
    page_map = []
    print(f"Extracting text from '{filename}' using Azure Form Recognizer")

    form_recognizer_client = DocumentAnalysisClient(
        endpoint=f"https://{args.formrecognizerservice}.cognitiveservices.azure.com/",
        credential=formrecognizer_creds,
        headers={"x-ms-useragent": "azure-search-resultcontentchat-demo/1.0.0"},
    )

    with open(filename, "rb") as f:
        poller = form_recognizer_client.begin_analyze_document(
            "prebuilt-layout", document=f
        )
    form_recognizer_results = poller.result()

    for page_num, page in enumerate(form_recognizer_results.pages):
        tables_on_page = [
            table
            for table in form_recognizer_results.tables
            if table.bounding_regions[0].page_number == page_num + 1
        ]

        # mark all positions of the table spans in the page
        page_offset = page.spans[0].offset
        page_length = page.spans[0].length
        table_chars = [-1] * page_length
        for table_id, table in enumerate(tables_on_page):
            for span in table.spans:
                # replace all table spans with "table_id" in table_chars array
                for i in range(span.length):
                    idx = span.offset - page_offset + i
                    if idx >= 0 and idx < page_length:
                        table_chars[idx] = table_id

        # build page text by replacing charcters in table spans with table html
        page_text = ""
        added_tables = set()
        for idx, table_id in enumerate(table_chars):
            if table_id == -1:
                page_text += form_recognizer_results.content[page_offset + idx]
            elif not table_id in added_tables:
                page_text += table_to_html(tables_on_page[table_id])
                added_tables.add(table_id)

        page_text += " "
        page_map.append((page_num, offset, page_text))
        offset += len(page_text)

    return page_map


def create_search_index():
    print(f"Ensuring search index {args.index} exists")
    index_client = SearchIndexClient(
        endpoint=f"https://{args.searchservice}.search.windows.net/",
        credential=search_creds,
    )
    if args.index not in index_client.list_index_names():
        index = SearchIndex(
            name=args.index,
            fields=[
                SimpleField(name="id", type="Edm.String", key=True),
                SearchableField(
                    name="content", type="Edm.String", analyzer_name="en.microsoft"
                ),
                SearchableField(
                    name="dni", type="Edm.String", analyzer_name="en.microsoft"
                ),
                SearchableField(
                    name="cuit", type="Edm.String", analyzer_name="en.microsoft"
                ),
                SearchableField(
                    name="npoliza", type="Edm.String", analyzer_name="en.microsoft"
                ),
                SimpleField(
                    name="category", type="Edm.String", filterable=True, facetable=True
                ),
                SimpleField(
                    name="sourcepage",
                    type="Edm.String",
                    filterable=True,
                    facetable=True,
                ),
                SimpleField(
                    name="sourcefile",
                    type="Edm.String",
                    filterable=True,
                    facetable=True,
                ),
            ],
            semantic_settings=SemanticSettings(
                configurations=[
                    SemanticConfiguration(
                        name="default",
                        prioritized_fields=PrioritizedFields(
                            title_field=None,
                            prioritized_content_fields=[
                                SemanticField(field_name="content")
                            ],
                        ),
                    )
                ]
            ),
        )
        print(f"Creating {args.index} search index")
        index_client.create_index(index)
    else:
        print(f"Search index {args.index} already exists")


def blob_name_from_file_page(filename, page=0):
    if os.path.splitext(filename)[1].lower() == ".pdf":
        return os.path.splitext(os.path.basename(filename))[0] + f"-{page}" + ".pdf"
    else:
        return os.path.basename(filename)


def upload_blobs(filename):
    blob_service = BlobServiceClient(
        account_url=f"https://{args.storageaccount}.blob.core.windows.net",
        credential=storage_creds,
    )
    blob_container = blob_service.get_container_client(args.container)
    if not blob_container.exists():
        blob_container.create_container()

    # if file is PDF split into pages and upload each page as a separate blob
    if os.path.splitext(filename)[1].lower() == ".pdf":
        reader = PdfReader(filename)
        pages = reader.pages
        for i in range(len(pages)):
            blob_name = blob_name_from_file_page(filename, i)
            print(f"\tUploading blob for page {i} -> {blob_name}")
            f = io.BytesIO()
            writer = PdfWriter()
            writer.add_page(pages[i])
            writer.write(f)
            f.seek(0)
            blob_container.upload_blob(blob_name, f, overwrite=True)
    else:
        blob_name = blob_name_from_file_page(filename)
        with open(filename, "rb") as data:
            blob_container.upload_blob(blob_name, data, overwrite=True)


def create_sections(filename, page_map):
    dni_value = None
    cuit_value = None
    npoliza_value = None
    sections = []

    for i, (section, pagenum, dni, cuit, npoliza) in enumerate(split_text(page_map)):
        # Almacenar los valores de DNI, CUIT y número de póliza para concatenación
        if dni:
            dni_value = dni
        if cuit:
            cuit_value = cuit
        if npoliza:
            npoliza_value = npoliza

        # Concatenar los valores de DNI, CUIT y número de póliza con la sección de contenido
        if dni_value or cuit_value or npoliza_value:
            concatenated_section = section
            if dni_value:
                concatenated_section = f"/DNI: {dni_value}/ {concatenated_section}"
            if cuit_value:
                concatenated_section = f"/CUIT: {cuit_value}/ {concatenated_section}"
            if npoliza_value:
                concatenated_section = (
                    f"/Npoliza: {npoliza_value}/ {concatenated_section}"
                )

            sections.append(
                {
                    "id": re.sub("[^0-9a-zA-Z_-]", "_", f"{filename}-{i}"),
                    "content": concatenated_section,
                    "category": args.category,
                    "sourcepage": blob_name_from_file_page(filename, pagenum),
                    "sourcefile": filename,
                    "dni": dni,
                    "cuit": cuit,
                    "npoliza": npoliza,
                }
            )

    return sections


def index_sections(filename, sections):
    print(f"Indexing sections from '{filename}' into search index '{args.index}'")
    search_client = SearchClient(
        endpoint=f"https://{args.searchservice}.search.windows.net/",
        index_name=args.index,
        credential=search_creds,
    )
    i = 0
    batch = []
    dni_value = None
    cuit_value = None
    npoliza_value = None
    for s in sections:
        batch.append(s)
        i += 1
        if s["dni"]:
            dni_value = s["dni"]
        if s["cuit"]:
            cuit_value = s["cuit"]
        if s["npoliza"]:
            npoliza_value = s[
                "npoliza"
            ]  # Asigna los valores si están presente en la sección
        if i % 1000 == 0:
            results = search_client.upload_documents(documents=batch)
            succeeded = sum([1 for r in results if r.succeeded])
            print(f"\tIndexed {len(results)} sections, {succeeded} succeeded")
            batch = []

    # Asigna el valor del DNI, del CUIT y de npoliza a todas las secciones del mismo archivo
    for s in batch:
        if dni_value:
            s["dni"] = dni_value
        if cuit_value:
            s["cuit"] = cuit_value
        if npoliza_value:
            s["npoliza"] = npoliza_value

    if len(batch) > 0:
        results = search_client.upload_documents(documents=batch)
        succeeded = sum([1 for r in results if r.succeeded])
        print(f"\tIndexed {len(results)} sections, {succeeded} succeeded")


def split_text(page_map):
    SENTENCE_ENDINGS = [".", "!", "?"]
    WORDS_BREAKS = [",", ";", ":", " ", "(", ")", "[", "]", "{", "}", "\t", "\n"]
    print(f"Splitting '{filename}' into sections")

    def find_page(offset):
        l = len(page_map)
        for i in range(l - 1):
            if offset >= page_map[i][1] and offset < page_map[i + 1][1]:
                return i
        return l - 1

    all_text = "".join(p[2] for p in page_map)
    length = len(all_text)
    start = 0
    end = length
    while start + SECTION_OVERLAP < length:
        last_word = -1
        end = start + MAX_SECTION_LENGTH

        if end > length:
            end = length
        else:
            # Try to find the end of the sentence
            while (
                end < length
                and (end - start - MAX_SECTION_LENGTH) < SENTENCE_SEARCH_LIMIT
                and all_text[end] not in SENTENCE_ENDINGS
            ):
                if all_text[end] in WORDS_BREAKS:
                    last_word = end
                end += 1
            if end < length and all_text[end] not in SENTENCE_ENDINGS and last_word > 0:
                end = last_word  # Fall back to at least keeping a whole word
        if end < length:
            end += 1

        # Try to find the start of the sentence or at least a whole word boundary
        last_word = -1
        while (
            start > 0
            and start > end - MAX_SECTION_LENGTH - 2 * SENTENCE_SEARCH_LIMIT
            and all_text[start] not in SENTENCE_ENDINGS
        ):
            if all_text[start] in WORDS_BREAKS:
                last_word = start
            start -= 1
        if all_text[start] not in SENTENCE_ENDINGS and last_word > 0:
            start = last_word
        if start > 0:
            start += 1

        section_text = all_text[start:end]
        yield (
            section_text,
            find_page(start),
            extract_dni(section_text),
            extract_cuit(section_text),
            extract_npoliza(section_text),
        )

        last_table_start = section_text.rfind("<table")
        if (
            last_table_start > 2 * SENTENCE_SEARCH_LIMIT
            and last_table_start > section_text.rfind("</table")
        ):
            # If the section ends with an unclosed table, we need to start the next section with the table.
            # If table starts inside SENTENCE_SEARCH_LIMIT, we ignore it, as that will cause an infinite loop for tables longer than MAX_SECTION_LENGTH
            # If last table starts inside SECTION_OVERLAP, keep overlapping
            print(
                f"Section ends with unclosed table, starting next section with the table at page {find_page(start)} offset {start} table start {last_table_start}"
            )
            start = min(end - SECTION_OVERLAP, start + last_table_start)
        else:
            start = end - SECTION_OVERLAP

    if start + SECTION_OVERLAP < end:
        yield (all_text[start:end], find_page(start))


create_search_index()
for filename in glob.glob(DATA_PATH):
    print("Processing:", filename)
    upload_blobs(filename)
    page_map = get_document_text(filename)
    dnis = extract_dni
    cuits = extract_cuit
    npolizas = extract_npoliza
    sections = create_sections(os.path.basename(filename), page_map)
    index_sections(os.path.basename(filename), sections)
