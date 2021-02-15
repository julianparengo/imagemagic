import scribus
import os
import sys

def main(argv):
    print("start : ")
    scribus.openDoc(argv[1])
    pdf = scribus.PDFfile()

    scribus.defineColor("CutContour", 0, 255, 0, 0)
    scribus.setSpotColor("CutContour", 1)

    scribus.replaceColor("FromPDF#000000", "CutContour")

    scribus.deleteColor("FromPDF#000000", "CutContour")

    pdf.version = 1.4
    pdf.file = argv[2]
    pdf.outdst = 1
    pdf.save()

if __name__ == '__main__':
    main(sys.argv)
