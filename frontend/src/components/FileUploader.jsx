import { useState } from 'react';
import { Button, Modal, FileInput, Alert, Progress } from '@mantine/core';
import { IconUpload, IconInfoCircle } from '@tabler/icons-react';
import { uploadFile } from '../provider';


function FileUploader({ refreshDatasets }) {
    const [file, setFile] = useState(undefined);
    const [progress, setProgress] = useState(0);
    const [status, setStatus] = useState('waiting');
    const [modalOpen, setModalOpen] = useState(false);

    const handleUpload = () => {
        setStatus('uploading');
        uploadFile(file, setProgress).then(() => {
            refreshDatasets();
            setProgress(0);
            setStatus('successful');
        }).catch(() => {
            setProgress(0);
            setStatus('error');
        })
    }

    return (
        <>
            <Button mt="24px" leftSection={<IconUpload size={14} />} variant="light" onClick={() => setModalOpen(true)}>
                Upload
            </Button>

            <Modal
                opened={modalOpen}
                onClose={() => {
                    // reset everything
                    setModalOpen(false);
                    setFile(undefined);
                    setProgress(0);
                    setStatus('waiting');
                }}
                title="Upload your file"
                centered
            >
                {status == 'successful' && (
                    <Alert variant="light" color="blue" title="File successfully uploaded." icon={<IconInfoCircle />} mb="sm" />
                )}
                {status == 'error' && (
                    <Alert variant="light" color="red" title="There was an issue uploading the file." icon={<IconInfoCircle />} mb="sm" />
                )}

                <FileInput
                    label="Upload file"
                    placeholder="Choose a file"
                    value={file}
                    onChange={setFile}
                    disabled={status == 'uploading'}
                />
                <Button mt="sm" leftSection={<IconUpload size={14} />} variant="light" onClick={handleUpload} disabled={!file || status == 'uploading'}>
                    Upload
                </Button>
                {status == 'uploading' && (
                    <Progress value={progress} mt="sm" />
                )}
            </Modal>
        </>
    );
}

export default FileUploader;
