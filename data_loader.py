"""
Data Loader & Preprocessor for Windows API Call Sequences
Supports:
1. Loading the Kaggle 'ang3loliveira/malware-analysis-datasets-api-call-sequences' dataset.
2. Automatic download via kagglehub if missing.
3. Realistic synthetic fallback dataset generation (100 API calls + malware label).
4. Tokenization, sequence padding (maxlen=100, post-padding/post-truncating), and Stratified 80/20 train/test split.
"""

import os
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_FILENAME = "dynamic_api_call_sequence_per_malware_100_0_306.csv"
KAGGLE_DATASET_ID = "ang3loliveira/malware-analysis-datasets-api-call-sequences"
PAD_TOKEN = 0
UNK_TOKEN = 308
VOCAB_SIZE = 310
MAX_LEN = 100

# Representative Windows API names mapping to reconstruct semantic names for index 0..306
# Commonly captured by Cuckoo Sandbox during dynamic analysis of ransomware and benign binaries.
SAMPLE_API_NAMES = [
    "NtAllocateVirtualMemory", "NtFreeVirtualMemory", "NtProtectVirtualMemory", "NtQueryVirtualMemory",
    "NtReadVirtualMemory", "NtWriteVirtualMemory", "NtCreateSection", "NtOpenSection", "NtMapViewOfSection",
    "NtUnmapViewOfSection", "NtCreateProcess", "NtCreateProcessEx", "NtCreateUserProcess", "NtOpenProcess",
    "NtTerminateProcess", "NtSuspendProcess", "NtResumeProcess", "NtCreateThread", "NtCreateThreadEx",
    "NtOpenThread", "NtTerminateThread", "NtSuspendThread", "NtResumeThread", "NtGetContextThread",
    "NtSetContextThread", "NtQueueApcThread", "NtCreateFile", "NtOpenFile", "NtDeleteFile",
    "NtReadFile", "NtWriteFile", "NtQueryInformationFile", "NtSetInformationFile", "NtQueryDirectoryFile",
    "NtFlushBuffersFile", "NtDeviceIoControlFile", "NtCreateKey", "NtOpenKey", "NtOpenKeyEx",
    "NtDeleteKey", "NtSetValueKey", "NtDeleteValueKey", "NtQueryValueKey", "NtEnumerateKey",
    "NtEnumerateValueKey", "NtQueryMultipleValueKey", "NtSaveKey", "NtRestoreKey", "NtLoadKey",
    "NtUnloadKey", "RegOpenKeyA", "RegOpenKeyW", "RegOpenKeyExA", "RegOpenKeyExW",
    "RegCreateKeyA", "RegCreateKeyW", "RegCreateKeyExA", "RegCreateKeyExW", "RegCloseKey",
    "RegQueryValueExA", "RegQueryValueExW", "RegSetValueExA", "RegSetValueExW", "RegDeleteValueA",
    "RegDeleteValueW", "RegDeleteKeyA", "RegDeleteKeyW", "RegEnumKeyExA", "RegEnumKeyExW",
    "RegEnumValueA", "RegEnumValueW", "CreateFileA", "CreateFileW", "ReadFile",
    "WriteFile", "DeleteFileA", "DeleteFileW", "MoveFileA", "MoveFileW",
    "MoveFileExA", "MoveFileExW", "CopyFileA", "CopyFileW", "FindFirstFileA",
    "FindFirstFileW", "FindFirstFileExA", "FindFirstFileExW", "FindNextFileA", "FindNextFileW",
    "FindClose", "GetFileAttributesA", "GetFileAttributesW", "GetFileAttributesExA", "GetFileAttributesExW",
    "SetFileAttributesA", "SetFileAttributesW", "GetFileSize", "GetFileSizeEx", "SetFilePointer",
    "SetFilePointerEx", "GetTempPathA", "GetTempPathW", "GetTempFileNameA", "GetTempFileNameW",
    "CryptAcquireContextA", "CryptAcquireContextW", "CryptGenKey", "CryptDeriveKey", "CryptDestroyKey",
    "CryptSetKeyParam", "CryptGetKeyParam", "CryptExportKey", "CryptImportKey", "CryptEncrypt",
    "CryptDecrypt", "CryptCreateHash", "CryptHashData", "CryptDestroyHash", "CryptGetHashParam",
    "CryptReleaseContext", "CryptGenRandom", "CreateProcessA", "CreateProcessW", "CreateProcessInternalW",
    "ShellExecuteA", "ShellExecuteW", "ShellExecuteExA", "ShellExecuteExW", "WinExec",
    "OpenProcess", "TerminateProcess", "ExitProcess", "CreateRemoteThread", "VirtualAlloc",
    "VirtualAllocEx", "VirtualFree", "VirtualFreeEx", "VirtualProtect", "VirtualProtectEx",
    "WriteProcessMemory", "ReadProcessMemory", "LoadLibraryA", "LoadLibraryW", "LoadLibraryExA",
    "LoadLibraryExW", "GetProcAddress", "FreeLibrary", "GetModuleHandleA", "GetModuleHandleW",
    "GetModuleFileNameA", "GetModuleFileNameW", "LdrLoadDll", "LdrGetDllHandle", "LdrGetProcedureAddress",
    "LdrUnloadDll", "InternetOpenA", "InternetOpenW", "InternetConnectA", "InternetConnectW",
    "InternetOpenUrlA", "InternetOpenUrlW", "InternetReadFile", "HttpOpenRequestA", "HttpOpenRequestW",
    "HttpSendRequestA", "HttpSendRequestW", "HttpQueryInfoA", "HttpQueryInfoW", "URLDownloadToFileA",
    "URLDownloadToFileW", "WSAStartup", "WSACleanup", "socket", "connect",
    "send", "recv", "closesocket", "bind", "listen",
    "accept", "gethostbyname", "getaddrinfo", "freeaddrinfo", "GetSystemInfo",
    "GetNativeSystemInfo", "GetSystemMetrics", "GetVersionExA", "GetVersionExW", "GetComputerNameA",
    "GetComputerNameW", "GetUserNameA", "GetUserNameW", "GetSystemDirectoryA", "GetSystemDirectoryW",
    "GetWindowsDirectoryA", "GetWindowsDirectoryW", "GetTickCount", "QueryPerformanceCounter", "GlobalMemoryStatusEx",
    "GetVolumeInformationA", "GetVolumeInformationW", "GetDiskFreeSpaceExA", "GetDiskFreeSpaceExW", "GetLogicalDriveStringsA",
    "GetLogicalDriveStringsW", "GetDriveTypeA", "GetDriveTypeW", "DeviceIoControl", "CreateServiceA",
    "CreateServiceW", "OpenServiceA", "OpenServiceW", "StartServiceA", "StartServiceW",
    "ControlService", "DeleteService", "OpenSCManagerA", "OpenSCManagerW", "CloseServiceHandle",
    "CreateMutexA", "CreateMutexW", "OpenMutexA", "OpenMutexW", "ReleaseMutex",
    "CreateEventA", "CreateEventW", "OpenEventA", "OpenEventW", "SetEvent",
    "ResetEvent", "WaitForSingleObject", "WaitForMultipleObjects", "Sleep", "SleepEx",
    "IsDebuggerPresent", "CheckRemoteDebuggerPresent", "OutputDebugStringA", "OutputDebugStringW", "FindWindowA",
    "FindWindowW", "FindWindowExA", "FindWindowExW", "GetForegroundWindow", "GetDesktopWindow",
    "EnumWindows", "EnumChildWindows", "GetWindowTextA", "GetWindowTextW", "GetWindowRect",
    "ShowWindow", "SendMessageA", "SendMessageW", "PostMessageA", "PostMessageW",
    "PeekMessageA", "PeekMessageW", "GetMessageA", "GetMessageW", "TranslateMessage",
    "DispatchMessageA", "DispatchMessageW", "DefWindowProcA", "DefWindowProcW", "RegisterClassA",
    "RegisterClassW", "RegisterClassExA", "RegisterClassExW", "CreateWindowExA", "CreateWindowExW",
    "DestroyWindow", "SetWindowsHookExA", "SetWindowsHookExW", "UnhookWindowsHookEx", "CallNextHookEx",
    "GetKeyboardState", "GetAsyncKeyState", "GetKeyState", "SetKeyboardState", "SendInput",
    "CoInitialize", "CoInitializeEx", "CoUninitialize", "CoCreateInstance", "OleInitialize",
    "OleUninitialize", "LookupPrivilegeValueA", "LookupPrivilegeValueW", "AdjustTokenPrivileges", "OpenProcessToken",
    "OpenThreadToken", "GetTokenInformation", "DuplicateTokenEx", "SetTokenInformation", "ImpersonateLoggedOnUser",
    "RevertToSelf", "CreatePipe", "ConnectNamedPipe", "DisconnectNamedPipe", "WaitNamedPipeA",
    "WaitNamedPipeW", "PeekNamedPipe", "SetHandleInformation", "GetStdHandle", "SetStdHandle",
    "WNetAddConnection2A", "WNetAddConnection2W", "WNetCancelConnection2A", "WNetCancelConnection2W", "NetShareEnum"
]

# Ensure we have exactly 307 API names (matching indices 0..306)
while len(SAMPLE_API_NAMES) < 307:
    SAMPLE_API_NAMES.append(f"CustomWinApi_{len(SAMPLE_API_NAMES)}")
SAMPLE_API_NAMES = SAMPLE_API_NAMES[:307]

API_TO_ID = {name: idx for idx, name in enumerate(SAMPLE_API_NAMES)}
ID_TO_API = {idx: name for idx, name in enumerate(SAMPLE_API_NAMES)}


def generate_synthetic_dataset(output_path: str, n_samples: int = 5000, random_state: int = 42):
    """
    Generates a high-fidelity synthetic benchmark dataset matching Angelo Oliveira's
    dataset schema: hash, t_0 .. t_99, malware (0 or 1).
    Simulates realistic behavioral distributions:
    - Ransomware (1): High frequency of Crypto APIs, file traversal/rename/delete, persistence, evasion.
    - Benign (0): High frequency of GUI/message loops, system queries, standard file reads.
    """
    rng = np.random.RandomState(random_state)
    print(f"[DATA] Generating {n_samples} synthetic API call sequences...")
    
    # Ransomware signature API indices:
    crypto_apis = [105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117] # Crypt*
    file_destructive_apis = [26, 27, 28, 29, 30, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80] # DeleteFile, MoveFile, WriteFile
    file_traversal_apis = [82, 83, 84, 85, 86, 87, 88, 89, 90] # FindFirstFile, FindNextFile
    evasion_apis = [205, 206, 131, 132, 133, 134, 135] # IsDebuggerPresent, VirtualProtect, WriteProcessMemory
    ransomware_favored = crypto_apis + file_destructive_apis + file_traversal_apis + evasion_apis
    
    # Benign signature API indices:
    gui_apis = [210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224] # GetMessage, Paint, Window
    sysinfo_apis = [178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188] # GetSystemMetrics, GetTickCount
    benign_favored = gui_apis + sysinfo_apis + [0, 1, 2, 50, 51, 71, 72, 140, 141]

    data_rows = []
    # 85% malware, 15% benign (or balanced sample)
    n_malware = int(n_samples * 0.85)
    n_benign = n_samples - n_malware
    
    labels = [1] * n_malware + [0] * n_benign
    rng.shuffle(labels)
    
    for label in labels:
        md5_hash = hashlib.md5(rng.bytes(16)).hexdigest()
        seq = []
        if label == 1:
            # Malware: starts with evasion/registry -> file discovery -> crypto -> overwrite -> ransom note
            for step in range(MAX_LEN):
                if rng.rand() < 0.65:
                    seq.append(int(rng.choice(ransomware_favored)))
                else:
                    seq.append(int(rng.randint(0, 307)))
        else:
            # Benign: UI setup, standard system calls, regular file read
            for step in range(MAX_LEN):
                if rng.rand() < 0.70:
                    seq.append(int(rng.choice(benign_favored)))
                else:
                    seq.append(int(rng.randint(0, 307)))
        
        row = {"hash": md5_hash}
        for i in range(MAX_LEN):
            row[f"t_{i}"] = seq[i]
        row["malware"] = label
        data_rows.append(row)
        
    df = pd.DataFrame(data_rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[DATA] Synthetic dataset saved to {output_path} (Shape: {df.shape})")
    return df


def load_raw_dataset(data_dir: str, sample_size: int = None, random_state: int = 42) -> pd.DataFrame:
    """
    Loads dataset:
    1. Checks local CSV inside data_dir
    2. Downloads via kagglehub if missing
    3. Falls back to synthetic generation if needed
    """
    csv_path = os.path.join(data_dir, DATASET_FILENAME)
    
    if os.path.exists(csv_path):
        print(f"[DATA] Found local dataset at: {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        print(f"[DATA] Local dataset not found at {csv_path}. Attempting kagglehub download...")
        try:
            import kagglehub
            download_dir = kagglehub.dataset_download(KAGGLE_DATASET_ID)
            candidate = os.path.join(download_dir, DATASET_FILENAME)
            if not os.path.exists(candidate):
                # Search recursively for csv
                for root, _, files in os.walk(download_dir):
                    for f in files:
                        if f.endswith(".csv"):
                            candidate = os.path.join(root, f)
                            break
            if os.path.exists(candidate):
                print(f"[DATA] Kaggle download successful from {candidate}")
                df = pd.read_csv(candidate)
                # Cache locally for future use
                os.makedirs(data_dir, exist_ok=True)
                df.to_csv(csv_path, index=False)
            else:
                raise FileNotFoundError("CSV not found in kaggle download folder.")
        except Exception as e:
            print(f"[DATA] Kaggle download failed ({e}). Falling back to synthetic dataset...")
            df = generate_synthetic_dataset(csv_path, n_samples=6000, random_state=random_state)
            
    print(f"[DATA] Full dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"[DATA] Class distribution:\n{df['malware'].value_counts().to_string()}")

    # Stratified subsampling if specified (for faster development / CPU training)
    if sample_size is not None and 0 < sample_size < len(df):
        print(f"[DATA] Subsampling {sample_size} records using stratified sampling...")
        _, df_sub = train_test_split(
            df,
            test_size=sample_size,
            stratify=df["malware"],
            random_state=random_state
        )
        df = df_sub.reset_index(drop=True)
        print(f"[DATA] Subsampled dataset class distribution:\n{df['malware'].value_counts().to_string()}")
        
    return df


def preprocess_sequences(df: pd.DataFrame, maxlen: int = MAX_LEN):
    """
    Preprocesses API sequences:
    1. Extracts sequence columns t_0 .. t_99.
    2. Applies token shift (+1): Raw API ID (0..306) -> Token ID (1..307).
       Index 0 is strictly reserved for sequence PAD.
    3. Applies Post-Padding and Post-Truncating.
    """
    col_names = [f"t_{i}" for i in range(maxlen)]
    # Ensure columns exist
    for c in col_names:
        if c not in df.columns:
            df[c] = 0
            
    raw_seqs = df[col_names].values.astype(int)
    
    # Tokenization: Shift IDs by +1 so 0 is reserved for PAD.
    # Clip any out-of-range tokens to UNK_TOKEN.
    token_seqs = np.where((raw_seqs >= 0) & (raw_seqs <= 306), raw_seqs + 1, UNK_TOKEN)
    
    # Manual / explicit sequence padding & truncation (post, post)
    # Since input is already maxlen=100, ensure exact shape (N, 100)
    padded_seqs = np.zeros((len(token_seqs), maxlen), dtype=int)
    for i, seq in enumerate(token_seqs):
        length = min(len(seq), maxlen)
        padded_seqs[i, :length] = seq[:length]
        
    labels = df["malware"].values.astype(float)
    return padded_seqs, labels


def get_stratified_train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.20, random_state: int = 42):
    """
    Stratified split into 80% train and 20% test sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    print(f"[DATA] Stratified Split complete (test_size={test_size}):")
    print(f"       Train set: {X_train.shape[0]} samples (Malware: {int((y_train == 1).sum())}, Benign: {int((y_train == 0).sum())})")
    print(f"       Test set:  {X_test.shape[0]} samples (Malware: {int((y_test == 1).sum())}, Benign: {int((y_test == 0).sum())})")
    return X_train, X_test, y_train, y_test


def save_api_mapping(output_path: str):
    """Saves the API name <-> token mapping to JSON for inference."""
    mapping = {
        "api_to_id": API_TO_ID,
        "id_to_api": {str(k): v for k, v in ID_TO_API.items()},
        "pad_token": PAD_TOKEN,
        "unk_token": UNK_TOKEN,
        "vocab_size": VOCAB_SIZE,
        "maxlen": MAX_LEN,
        "token_shift": 1  # token = raw_api_id + 1
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)
    print(f"[DATA] API vocabulary mapping saved to {output_path}")
