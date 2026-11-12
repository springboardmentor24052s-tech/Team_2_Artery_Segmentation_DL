const DB_NAME = 'arteryai-scan-assets';
const STORE_NAME = 'scans';
const DB_VERSION = 1;

function openDatabase() {
  return new Promise((resolve, reject) => {
    if (!('indexedDB' in window)) {
      reject(new Error('This browser does not support local scan preview storage.'));
      return;
    }
    const request = window.indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: 'caseId' });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error || new Error('Could not open scan storage.'));
  });
}

export async function saveScanAsset(caseId, file) {
  const extension = file.name.split('.').pop()?.toLowerCase();
  const inferredType = ({ png: 'image/png', jpg: 'image/jpeg', jpeg: 'image/jpeg', tif: 'image/tiff', tiff: 'image/tiff' })[extension] || file.type || '';
  const db = await openDatabase();
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(STORE_NAME, 'readwrite');
    transaction.objectStore(STORE_NAME).put({ caseId, blob: file, name: file.name, type: inferredType, savedAt: new Date().toISOString() });
    transaction.oncomplete = () => { db.close(); resolve(true); };
    transaction.onerror = () => { db.close(); reject(transaction.error || new Error('Could not save scan preview.')); };
    transaction.onabort = () => { db.close(); reject(transaction.error || new Error('Scan preview storage was interrupted.')); };
  });
}

export async function getScanAsset(caseId) {
  const db = await openDatabase();
  return new Promise((resolve, reject) => {
    const request = db.transaction(STORE_NAME, 'readonly').objectStore(STORE_NAME).get(caseId);
    request.onsuccess = () => { db.close(); resolve(request.result || null); };
    request.onerror = () => { db.close(); reject(request.error || new Error('Could not retrieve scan preview.')); };
  });
}
