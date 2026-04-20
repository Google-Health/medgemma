# DICOMs for testing

## Source and processing

These DICOMs were taken from a subset of LIDC and then further de-identified by
removing the following tags and blanking out the images.

```
    StudyInstanceUID
    SeriesInstanceUID
    FrameOfReferenceUID
    SOPInstanceUID
```

```
    (0x0010, 0x0010),  # PatientName
    (0x0010, 0x0020),  # PatientID
    (0x0010, 0x0030),  # PatientBirthDate
    (0x0008, 0x0060),  # Modality
    (0x0008, 0x1030),  # StudyDescription
    (0x0008, 0x0050),  # AccessionNumber
    (0x0008, 0x103e),  # SeriesDescription
    (0x0008, 0x0080),  # InstitutionName
    (0x0008, 0x0070),  # Manufacturer
    (0x0008, 0x1090),  # Manufacturer's Model Name
    (0x0008, 0x0081),  # InstitutionAddress
    (0x0008, 0x1040),  # Institutional Department Name
    (0x0008, 0x1010),  # StationName
    (0x0010, 0x1000),  # Other Patient IDs
    (0x0010, 0x1010),  # Patient's Age
    (0x0010, 0x1020),  # Patient's Size
    (0x0010, 0x1030),  # Patient's Weight
    (0x0010, 0x21b0),  # Additional Patient History
    (0x0032, 0x1032),  # Requesting Physician
    (0x0032, 0x1060),  # Requested Procedure Description
    (0x0038, 0x0010),  # Admission ID
    (0x0038, 0x0020),  # Admitting Date
    (0x0038, 0x0021),  # Admitting Time
    (0x0018, 0x0010), # Bolus
    (0x0018, 0x1030), # protocol name
    (0x0008, 0x0005),  # Specific Character Set (use with caution)
    (0x0008, 0x0008),  # Image Type
    (0x0008, 0x0020),  # Study Date
    (0x0008, 0x0021),  # Series Date
    (0x0008, 0x0022),  # Acquisition Date
    (0x0008, 0x0023),  # Content Date
    (0x0008, 0x0024),  # Overlay Date
    (0x0008, 0x0025),  # Curve Date
    (0x0008, 0x002a),  # Acquisition DateTime
    (0x0008, 0x0030),  # Study Time
    (0x0008, 0x0032),  # Acquisition Time
    (0x0008, 0x0033),  # Content Time
    (0x0008, 0x0090),  # Referring Physician's Name
    (0x0008, 0x1155),  # Referenced SOP Instance UID (use with caution)
    (0x0010, 0x21d0),  # Last Menstrual Date
    (0x0012, 0x0062),  # Patient Identity Removed
    (0x0012, 0x0063),  # De-identification Method
    (0x0018, 0x0022),  # Scan Options
    (0x0018, 0x0060),  # KVP
    (0x0018, 0x1030),  # Protocol Name
    (0x0018, 0x1040),  # Contrast/Bolus Route
    (0x0018, 0x1150),  # Exposure Time
    (0x0018, 0x1151),  # X-Ray Tube Current
    (0x0018, 0x1152),  # Exposure
    (0x0018, 0x1160),  # Filter Type
    (0x0018, 0x1170),  # Generator Power
    (0x0018, 0x1190),  # Focal Spot(s)
    (0x0018, 0x1210),  # Convolution Kernel
    (0x0020, 0x0010),  # Study ID
    (0x0020, 0x0012),  # Acquisition Number
    (0x0020, 0x1040),  # Position Reference Indicator
    (0x0028, 0x0303),  # Longitudinal Temporal Information Modified
    (0x0040, 0x0002),  # Scheduled Procedure Step Start Date
    (0x0040, 0x0004),  # Scheduled Procedure Step End Date
    (0x0040, 0x0244),  # Performed Procedure Step Start Date
    (0x0040, 0x2016),  # Placer Order Number / Imaging Service Request
    (0x0040, 0x2017),  # Filler Order Number / Imaging Service Request
    (0x0040, 0xa075),  # Verifying Observer Name
    (0x0040, 0xa123),  # Person Name
    (0x0040, 0xa124),  # UID
    (0x0070, 0x0084),  # Content Creator's Name
    (0x0088, 0x0140),  # Storage Media File-set UID
    (0x0002, 0x0000),  # File Meta Information Group Length
    (0x0002, 0x0001),  # File Meta Information Version
    (0x0002, 0x0012),  # Implementation Class UID
    (0x0002, 0x0013),  # Implementation Version Name
    (0x0002, 0x0013),  # Implementation Version
    (0x0018, 0x0015)
```
