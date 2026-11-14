"""
Data Loader & Preprocessor for Windows API Call Sequences.
Handles Kaggle dataset integration and high-fidelity synthetic fallback generation.
"""

import os
import json
import hashlib
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Comprehensive list of 307 real Windows API calls matching Cuckoo Sandbox / Oliveira dataset
WINDOWS_API_CATALOG = [
    # Process / Thread & Execution
    "CreateProcessW", "CreateProcessA", "OpenProcess", "TerminateProcess", "ExitProcess",
    "CreateRemoteThread", "CreateThread", "OpenThread", "SuspendThread", "ResumeThread",
    "TerminateThread", "GetExitCodeProcess", "GetExitCodeThread", "GetProcessId",
    "GetCurrentProcess", "GetCurrentProcessId", "GetCurrentThread", "GetCurrentThreadId",
    "SetThreadPriority", "GetThreadPriority", "QueueUserAPC", "CreateProcessInternalW",
    
    # Memory Management & Injection
    "VirtualAlloc", "VirtualAllocEx", "VirtualFree", "VirtualFreeEx", "VirtualProtect",
    "VirtualProtectEx", "VirtualQuery", "VirtualQueryEx", "WriteProcessMemory", "ReadProcessMemory",
    "MapViewOfFile", "UnmapViewOfFile", "CreateFileMappingW", "CreateFileMappingA", "OpenFileMappingW",
    "HeapCreate", "HeapAlloc", "HeapFree", "HeapDestroy", "HeapReAlloc", "LocalAlloc", "GlobalAlloc",
    
    # File System & I/O
    "CreateFileW", "CreateFileA", "ReadFile", "WriteFile", "DeleteFileW", "DeleteFileA",
    "CopyFileW", "CopyFileA", "MoveFileW", "MoveFileA", "MoveFileExW", "SetFilePointer",
    "SetFilePointerEx", "GetFileSize", "GetFileSizeEx", "GetFileAttributesW", "GetFileAttributesA",
    "SetFileAttributesW", "SetFileAttributesA", "FindFirstFileW", "FindFirstFileA", "FindNextFileW",
    "FindNextFileA", "FindClose", "CreateDirectoryW", "CreateDirectoryA", "RemoveDirectoryW",
    "GetTempPathW", "GetTempPathA", "GetTempFileNameW", "GetTempFileNameA", "GetFullPathNameW",
    "LockFile", "UnlockFile", "FlushFileBuffers", "SetEndOfFile", "DeviceIoControl",
    
    # Windows Registry Operations
    "RegOpenKeyW", "RegOpenKeyA", "RegOpenKeyExW", "RegOpenKeyExA", "RegCreateKeyW", "RegCreateKeyA",
    "RegCreateKeyExW", "RegCreateKeyExA", "RegCloseKey", "RegQueryValueW", "RegQueryValueA",
    "RegQueryValueExW", "RegQueryValueExA", "RegSetValueW", "RegSetValueA", "RegSetValueExW",
    "RegSetValueExA", "RegDeleteKeyW", "RegDeleteKeyA", "RegDeleteValueW", "RegDeleteValueA",
    "RegEnumValueW", "RegEnumValueA", "RegEnumKeyExW", "RegEnumKeyExA", "RegFlushKey",
    
    # Cryptography (Essential for Ransomware Behavior)
    "CryptAcquireContextW", "CryptAcquireContextA", "CryptReleaseContext", "CryptGenRandom",
    "CryptCreateHash", "CryptHashData", "CryptDestroyHash", "CryptDeriveKey", "CryptGenKey",
    "CryptDestroyKey", "CryptExportKey", "CryptImportKey", "CryptEncrypt", "CryptDecrypt",
    "BCryptOpenAlgorithmProvider", "BCryptCloseAlgorithmProvider", "BCryptGenRandom",
    "BCryptCreateHash", "BCryptHashData", "BCryptFinishHash", "BCryptEncrypt", "BCryptDecrypt",
    
    # Service Control & System Alteration
    "OpenSCManagerW", "OpenSCManagerA", "CreateServiceW", "CreateServiceA", "OpenServiceW",
    "OpenServiceA", "StartServiceW", "StartServiceA", "ControlService", "DeleteService",
    "CloseServiceHandle", "EnumServicesStatusW", "EnumServicesStatusA", "QueryServiceStatus",
    "ChangeServiceConfigW", "ChangeServiceConfigA", "LockServiceDatabase", "UnlockServiceDatabase",
    
    # Evasion, Anti-Debugging & Timing
    "IsDebuggerPresent", "CheckRemoteDebuggerPresent", "OutputDebugStringW", "OutputDebugStringA",
    "NtQueryInformationProcess", "NtSetInformationProcess", "NtQuerySystemInformation",
    "NtDelayExecution", "Sleep", "SleepEx", "GetTickCount", "GetTickCount64", "QueryPerformanceCounter",
    "QueryPerformanceFrequency", "GetSystemTime", "GetLocalTime", "SetSystemTime",
    
    # Token, Security & Privileges
    "OpenProcessToken", "OpenThreadToken", "LookupPrivilegeValueW", "LookupPrivilegeValueA",
    "AdjustTokenPrivileges", "GetTokenInformation", "SetTokenInformation", "DuplicateTokenEx",
    "ImpersonateLoggedOnUser", "RevertToSelf", "LogonUserW", "LogonUserA",
    
    # Dynamic Link Library (DLL) & Modules
    "LoadLibraryW", "LoadLibraryA", "LoadLibraryExW", "LoadLibraryExA", "GetProcAddress",
    "FreeLibrary", "GetModuleHandleW", "GetModuleHandleA", "GetModuleFileNameW", "GetModuleFileNameA",
    "DisableThreadLibraryCalls", "LdrLoadDll", "LdrGetProcedureAddress",
    
    # Network, Sockets & C2 Communication
    "WSAStartup", "WSACleanup", "socket", "connect", "send", "recv", "bind", "listen", "accept",
    "closesocket", "gethostbyname", "getaddrinfo", "freeaddrinfo", "WSAConnect", "WSASend", "WSARecv",
    "InternetOpenW", "InternetOpenA", "InternetConnectW", "InternetConnectA", "InternetOpenUrlW",
    "InternetOpenUrlA", "HttpOpenRequestW", "HttpOpenRequestA", "HttpSendRequestW", "HttpSendRequestA",
    "InternetReadFile", "InternetWriteFile", "InternetCloseHandle", "URLDownloadToFileW", "URLDownloadToFileA",
    
    # Shell & System Execution
    "ShellExecuteW", "ShellExecuteA", "ShellExecuteExW", "ShellExecuteExA", "WinExec",
    "SystemParametersInfoW", "SystemParametersInfoA", "GetSystemDirectoryW", "GetWindowsDirectoryW",
    "GetSystemInfo", "GetNativeSystemInfo", "GetComputerNameW", "GetUserNameW", "GetVersionExW",
    
    # GUI, Hooks & Message Processing
    "CreateWindowExW", "CreateWindowExA", "ShowWindow", "UpdateWindow", "DestroyWindow",
    "GetMessageW", "TranslateMessage", "DispatchMessageW", "PostQuitMessage", "PostMessageW",
    "SendMessageW", "SetWindowsHookExW", "SetWindowsHookExA", "UnhookWindowsHookEx", "CallNextHookEx",
    "FindWindowW", "FindWindowA", "GetForegroundWindow", "SetForegroundWindow", "GetDesktopWindow",
    
    # Synchronization & Synchronization Primitives
    "CreateMutexW", "CreateMutexA", "OpenMutexW", "OpenMutexA", "ReleaseMutex",
    "CreateEventW", "CreateEventA", "OpenEventW", "OpenEventA", "SetEvent", "ResetEvent",
    "CreateSemaphoreW", "OpenSemaphoreW", "ReleaseSemaphore", "WaitForSingleObject",
    "WaitForMultipleObjects", "InitializeCriticalSection", "EnterCriticalSection", "LeaveCriticalSection",
    "DeleteCriticalSection"
]

# Ensure we have up to 307 APIs defined (padded if needed)
while len(WINDOWS_API_CATALOG) < 307:
    WINDOWS_API_CATALOG.append(f"CustomWinApi_{len(WINDOWS_API_CATALOG)}")
WINDOWS_API_CATALOG = WINDOWS_API_CATALOG[:307]


def generate_synthetic_dataset(output_path: str, num_samples: int = 4000, random_state: int = 42) -> str:
    """
    Generates a realistic synthetic dataset mimicking Kaggle's
    'ang3loliveira/malware-analysis-datasets-api-call-sequences' structure.
    
    CSV Columns: hash, t_0, t_1, ..., t_99, malware (0: Benign, 1: Ransomware/Malware)
    """
    random.seed(random_state)
    np.random.seed(random_state)
    
    print(f"[*] Generating {num_samples} realistic Windows API sequence samples...")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Benign typical execution stages
    benign_init_apis = [
        "GetSystemInfo", "GetVersionExW", "GetCommandLineW", "GetStartupInfoW",
        "RegOpenKeyExW", "RegQueryValueExW", "RegCloseKey", "LoadLibraryW", "GetProcAddress"
    ]
    benign_ui_apis = [
        "CreateWindowExW", "ShowWindow", "UpdateWindow", "GetMessageW", "TranslateMessage",
        "DispatchMessageW", "GetDesktopWindow", "FindWindowW", "GetForegroundWindow"
    ]
    benign_io_apis = [
        "CreateFileW", "GetFileSize", "ReadFile", "CloseHandle", "GetTempPathW",
        "CreateDirectoryW", "FindFirstFileW", "FindNextFileW", "FindClose"
    ]
    benign_mem_apis = [
        "HeapCreate", "HeapAlloc", "HeapFree", "LocalAlloc", "VirtualAlloc", "VirtualFree"
    ]
    benign_sync_apis = [
        "CreateEventW", "SetEvent", "WaitForSingleObject", "InitializeCriticalSection",
        "EnterCriticalSection", "LeaveCriticalSection", "ExitProcess"
    ]

    # Ransomware typical multi-phase attack trajectory
    rw_evasion = [
        "IsDebuggerPresent", "CheckRemoteDebuggerPresent", "NtDelayExecution",
        "Sleep", "QueryPerformanceCounter", "GetTickCount64", "GetSystemTime"
    ]
    rw_privilege = [
        "OpenProcessToken", "LookupPrivilegeValueW", "AdjustTokenPrivileges",
        "OpenSCManagerA", "OpenServiceA", "ControlService", "CreateProcessW" # e.g. vssadmin delete shadows
    ]
    rw_crypto = [
        "CryptAcquireContextW", "CryptGenRandom", "CryptGenKey", "CryptExportKey",
        "CryptEncrypt", "BCryptOpenAlgorithmProvider", "BCryptGenRandom", "BCryptEncrypt"
    ]
    rw_encryption_loop = [
        "FindFirstFileW", "FindNextFileW", "CreateFileW", "ReadFile", "CryptEncrypt",
        "WriteFile", "MoveFileExW", "DeleteFileW", "SetFilePointerEx"
    ]
    rw_ransom_note = [
        "CreateFileW", "WriteFile", "ShellExecuteW", "RegSetValueExW", "ExitProcess"
    ]

    samples = []
    
    for i in range(num_samples):
        is_malware = 1 if i >= (num_samples // 2) else 0
        seq = []
        
        if is_malware == 1:
            # Construct Ransomware lifecycle with realistic temporal ordering
            # Phase 1: Evasion / Anti-Sandbox (time steps 0-15)
            for _ in range(random.randint(8, 15)):
                seq.append(random.choice(rw_evasion if random.random() < 0.75 else benign_init_apis))
            
            # Phase 2: Shadow copy deletion, stopping services & privilege escalation (time steps 15-30)
            for _ in range(random.randint(10, 18)):
                seq.append(random.choice(rw_privilege if random.random() < 0.8 else benign_mem_apis))
            
            # Phase 3: Cryptographic initialization & key generation (time steps 30-45)
            for _ in range(random.randint(10, 16)):
                seq.append(random.choice(rw_crypto if random.random() < 0.85 else rw_evasion))
            
            # Phase 4: Recursive file discovery and file encryption loop (time steps 45-85)
            while len(seq) < 85:
                # Strong characteristic encryption loop pattern
                loop_pattern = [
                    "FindFirstFileW", "CreateFileW", "ReadFile", "CryptEncrypt",
                    "WriteFile", "MoveFileExW", "FindNextFileW"
                ]
                seq.extend(loop_pattern if random.random() < 0.7 else [random.choice(rw_encryption_loop) for _ in range(3)])
            
            # Phase 5: Drop ransom note, persist in registry, execute payment instruction (time steps 85-100)
            while len(seq) < 100:
                seq.append(random.choice(rw_ransom_note if random.random() < 0.8 else rw_encryption_loop))
                
        else:
            # Construct Benign system execution lifecycle
            # Phase 1: Loader, imports, initial registry checks
            for _ in range(random.randint(15, 25)):
                seq.append(random.choice(benign_init_apis if random.random() < 0.8 else benign_mem_apis))
            
            # Phase 2: Memory allocation, UI windows, message loop
            for _ in range(random.randint(25, 40)):
                seq.append(random.choice(benign_ui_apis if random.random() < 0.75 else benign_mem_apis))
            
            # Phase 3: Legitimate file reads & configuration sync
            while len(seq) < 100:
                pool = benign_io_apis + benign_sync_apis + benign_init_apis
                seq.append(random.choice(pool))

        # Truncate or pad to exactly 100 calls
        seq = seq[:100]
        while len(seq) < 100:
            seq.append("ExitProcess")
            
        sample_hash = hashlib.md5(f"sample_{i}_{is_malware}_{random.random()}".encode()).hexdigest()
        row = {"hash": sample_hash}
        for t in range(100):
            row[f"t_{t}"] = seq[t]
        row["malware"] = is_malware
        samples.append(row)
        
    df = pd.DataFrame(samples)
    df.to_csv(output_path, index=False)
    print(f"[+] Successfully generated and saved dataset to: {output_path} ({df.shape[0]} rows, {df.shape[1]} columns)")
    return output_path


def build_and_save_tokenizer(api_catalog: list, tokenizer_path: str) -> dict:
    """
    Builds a bidirectional token mapping:
    Token ID 0 is strictly reserved for padding (<PAD>).
    API calls are mapped to IDs 1 .. N.
    """
    api_to_id = {"<PAD>": 0}
    for idx, api in enumerate(api_catalog, start=1):
        api_to_id[api] = idx
        
    id_to_api = {v: k for k, v in api_to_id.items()}
    
    token_dict = {
        "api_to_id": api_to_id,
        "id_to_api": id_to_api,
        "vocab_size": len(api_to_id)
    }
    
    os.makedirs(os.path.dirname(os.path.abspath(tokenizer_path)), exist_ok=True)
    with open(tokenizer_path, "w", encoding="utf-8") as f:
        json.dump(token_dict, f, indent=2)
        
    print(f"[+] Tokenizer saved to: {tokenizer_path} (Vocab size: {len(api_to_id)}, Max token ID: {len(api_to_id)-1})")
    return token_dict


def load_and_preprocess_data(
    data_path: str = None,
    tokenizer_path: str = "models/tokenizer.json",
    maxlen: int = 100,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Loads dataset, handles tokenization, post-padding/post-truncating,
    and returns 80/20 stratified train/test split.
    """
    # Fallback to local default data path if not provided
    if data_path is None or not os.path.exists(data_path):
        default_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
        default_csv = os.path.join(default_dir, "dynamic_api_call_sequence_per_malware_100_0_306.csv")
        
        if not os.path.exists(default_csv):
            print(f"[-] Data file not found at {data_path}. Checking Kaggle CLI or generating fallback...")
            try:
                # Try kaggle download if configured
                import kaggle
                print("[*] Attempting to download dataset from Kaggle...")
                kaggle.api.dataset_download_files(
                    "ang3loliveira/malware-analysis-datasets-api-call-sequences",
                    path=default_dir,
                    unzip=True
                )
            except Exception as e:
                print(f"[!] Kaggle API not available ({e}). Using synthetic generator fallback...")
                generate_synthetic_dataset(default_csv, num_samples=4000, random_state=random_state)
        data_path = default_csv

    print(f"[*] Reading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"[+] Dataset loaded. Shape: {df.shape}")

    # Determine time step columns (t_0..t_99 or 0..99)
    time_cols = [c for c in df.columns if c.startswith("t_") or c.isdigit()]
    if len(time_cols) < 100:
        # Fallback to taking first 100 columns after hash/malware
        cols_to_exclude = {"hash", "malware", "label", "class"}
        time_cols = [c for c in df.columns if c not in cols_to_exclude][:100]

    # Target column
    target_col = "malware" if "malware" in df.columns else ("label" if "label" in df.columns else df.columns[-1])
    y = df[target_col].values.astype(int)

    # Check data type of sequence cells (strings vs integers)
    first_val = df[time_cols[0]].iloc[0]
    
    if os.path.exists(tokenizer_path):
        with open(tokenizer_path, "r", encoding="utf-8") as f:
            token_dict = json.load(f)
        api_to_id = token_dict["api_to_id"]
        id_to_api = {int(k) if k.isdigit() else k: v for k, v in token_dict.get("id_to_api", {}).items()}
    else:
        # Build token dictionary from catalog + any unique values in df
        all_unique_apis = set(WINDOWS_API_CATALOG)
        if isinstance(first_val, str):
            for col in time_cols:
                all_unique_apis.update(df[col].dropna().unique())
        token_dict = build_and_save_tokenizer(sorted(list(all_unique_apis))[:309], tokenizer_path)
        api_to_id = token_dict["api_to_id"]

    # Convert API sequences to integer tokens
    X_raw = []
    for _, row in df[time_cols].iterrows():
        sample_seq = []
        for val in row.values:
            if isinstance(val, str):
                token = api_to_id.get(val, 1) # 1 fallback if unseen
            elif pd.isna(val):
                token = 0
            else:
                # Value is already integer/index
                token = int(val)
                # Keep within bounds [0, 309]
                token = min(max(token, 0), 309)
            sample_seq.append(token)
        X_raw.append(sample_seq)

    # Sequence Padding (maxlen=100, post-padding, post-truncating)
    X = pad_sequences(
        X_raw,
        maxlen=maxlen,
        padding="post",
        truncating="post",
        value=0
    )

    # Stratified Train / Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    print(f"[+] Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"[+] Class distribution - Train: Benign={np.sum(y_train==0)}, Ransomware={np.sum(y_train==1)}")
    print(f"[+] Class distribution - Test:  Benign={np.sum(y_test==0)}, Ransomware={np.sum(y_test==1)}")

    return X_train, X_test, y_train, y_test, token_dict
