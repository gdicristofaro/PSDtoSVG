export async function processFile(file: File) {
  const formData = new FormData();
  formData.append('psd_file', file);

  const response = await fetch('/api/v1/upload', {
    method: 'POST',
    body: formData
  });

  if (!response.ok) {
    throw new Error(`Upload failed with status ${response.status}`);
  }

  return await response.text();
}
