/**
 * Utility formatters for dates, file sizes, and identifiers
 */

export function formatDate(dateString) {
  if (!dateString) return '—';
  try {
    const date = new Date(dateString);
    if (isNaN(date.getTime())) return dateString;
    return new Intl.DateTimeFormat('en-US', {
      year: 'numeric',
      month: 'short',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
    }).format(date);
  } catch {
    return dateString;
  }
}

export function formatDateShort(dateString) {
  if (!dateString) return '—';
  try {
    const date = new Date(dateString);
    if (isNaN(date.getTime())) return dateString;
    return new Intl.DateTimeFormat('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    }).format(date);
  } catch {
    return dateString;
  }
}

export function formatBytes(bytes, decimals = 1) {
  if (bytes === 0) return '0 Bytes';
  if (!bytes || isNaN(bytes)) return '—';
  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(dm))} ${sizes[i]}`;
}

/**
 * Generate a synthetic ARCADE study identifier (e.g. ARC-2025-008)
 */
export function generateSyntheticCaseId(existingCases = []) {
  const currentYear = new Date().getFullYear();
  const maxNum = existingCases.reduce((max, c) => {
    const match = c.id && c.id.match(/ARC-\d{4}-(\d+)/);
    if (match) {
      const num = parseInt(match[1], 10);
      return num > max ? num : max;
    }
    return max;
  }, 0);

  const nextNum = String(maxNum + 1).padStart(3, '0');
  return `ARC-${currentYear}-${nextNum}`;
}
