import hashlib
import io
import pickle
from zipfile import ZipFile

import pytest

from src.scripts import download_model as downloader


def response(content, content_type='application/octet-stream'):
    stream = io.BytesIO(content)
    stream.headers = {'Content-Type': content_type}
    return stream


def test_download_model_drive_confirmation_and_reuse(tmp_path, monkeypatch):
    files = {
        'config.json': b'{}',
        'pytorch_model.bin': b'weights',
        'MLM-re_token_dictionary_v1.pkl': b'tokens',
        'mouse_gene_median_dictionary.pkl': b'medians',
        downloader.GENE_MAP_PICKLE: pickle.dumps({'Gene1': 'ENSMUSG001'}),
        'gene_map.csv': b'gene_symbol,ensembl_id\nGene1,ENSMUSG001\n',
    }
    archive = io.BytesIO()
    with ZipFile(archive, 'w') as handle:
        for name in ['config.json', 'pytorch_model.bin']:
            handle.writestr('mouse-Geneformer/' + name, files[name])
        handle.writestr('../unexpected.txt', 'must not be extracted')
    payload = archive.getvalue()
    monkeypatch.setattr(downloader, 'HASHES', {name: hashlib.sha256(data).hexdigest()
                                             for name, data in files.items()})
    monkeypatch.setattr(downloader, 'MODEL_SHA256', hashlib.sha256(payload).hexdigest())
    calls = []

    def open_url(request, timeout):
        url = request.full_url
        calls.append(url)
        if url == downloader.MODEL_URL:
            return response(b'<input name="confirm" value="t"><input name="uuid" value="test-id">', 'text/html')
        if url.startswith(downloader.DRIVE_URL):
            assert 'confirm=t' in url and 'uuid=test-id' in url
            return response(payload)
        return response(files[url.rsplit('/', 1)[1]])

    monkeypatch.setattr(downloader, 'urlopen', open_url)
    output = tmp_path / 'model'
    downloader.download_model(output)
    assert len(calls) == 5
    for name, data in files.items():
        assert (output / name).read_bytes() == data
    assert not (tmp_path / 'unexpected.txt').exists()
    assert (output / 'resources.json').exists()
    downloader.download_model(output)
    assert len(calls) == 5  # Reutiliza tudo, sem acesso à rede.
    (output / 'mouse_gene_median_dictionary.pkl').unlink()
    downloader.download_model(output)
    assert len(calls) == 6  # Recupera apenas o arquivo ausente.


def test_bad_download_is_not_published(tmp_path, monkeypatch):
    monkeypatch.setattr(downloader, 'urlopen', lambda *a, **kw: response(b'truncated'))
    destination = tmp_path / 'weights.bin'
    with pytest.raises(ValueError, match='SHA-256'):
        downloader.download_file('https://example.com/weights', destination, 'incorrect')
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


def test_existing_different_file_is_preserved(tmp_path, monkeypatch):
    destination = tmp_path / 'config.json'
    destination.write_text('local content')
    def no_network(*args, **kwargs):
        pytest.fail('Não deveria baixar com arquivo local divergente.')
    monkeypatch.setattr(downloader, 'urlopen', no_network)
    with pytest.raises(ValueError, match='existente difere'):
        downloader.download_model(tmp_path)
    assert destination.read_text() == 'local content'


def test_html_without_confirmation_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(downloader, 'urlopen', lambda *a, **kw: response(b'<html>Quota exceeded</html>', 'text/html'))
    with pytest.raises(ValueError, match='confirmação'):
        downloader.download_file(downloader.MODEL_URL, tmp_path / 'model.zip', 'unused')
    assert list(tmp_path.iterdir()) == []
