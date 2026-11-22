import { UploadIcon } from "lucide-react";
import { useDropzone } from "react-dropzone";

export const pdfFiles = { "application/pdf": [".pdf"] };
export const proposalFiles = {
  ...pdfFiles,
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [".docx"],
};

export default function Dropzone({ accept, file, onSelect, onError, disabled, compact = false }) {
  const label = compact ? "Upload a research proposal" : "Upload a paper";
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept,
    multiple: false,
    maxSize: 12 * 1024 * 1024,
    disabled,
    onDrop(accepted, rejected) {
      if (rejected.length) {
        onError(rejected[0].errors[0].message);
        return;
      }
      if (accepted.length) onSelect(accepted[0]);
    },
  });

  return (
    <button {...getRootProps({
      type: "button",
      disabled,
      "aria-label": label,
      title: compact ? label : undefined,
      className: `upload-zone ${compact ? "upload-compact" : ""} ${isDragActive ? "drag-active" : ""}`,
    })}>
      <input {...getInputProps()} />
      <span className="upload-glyph"><UploadIcon size={18} /></span>
      {!compact && <>
        <strong>{file?.name || label}</strong>
        <small>{file ? "Drag and drop or click to replace" : "Drag and drop or click to upload"}</small>
        <small>PDF · Up to 12 MB</small>
      </>}
    </button>
  );
}
