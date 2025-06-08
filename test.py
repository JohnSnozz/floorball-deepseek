import warnings
import logging

# Alle PDF-related warnings unterdrücken
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*CropBox.*")

# PyPDF2 logging unterdrücken
logging.getLogger("PyPDF2").setLevel(logging.CRITICAL)
logging.getLogger("pdfplumber").setLevel(logging.CRITICAL)


from ollamafunctions import orchestration


print(orchestration("Wer darf mit dem Schiri reden?"))
