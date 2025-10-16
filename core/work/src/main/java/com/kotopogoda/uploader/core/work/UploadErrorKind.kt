package com.kotopogoda.uploader.core.work

enum class UploadErrorKind(val rawValue: String) {
    NETWORK("network"),
    IO("io"),
    HTTP("http"),
    REMOTE_FAILURE("remoteFailure"),
    UNEXPECTED("unexpected");

    companion object {
        fun fromRawValue(raw: String?): UploadErrorKind? =
            raw?.let { values().firstOrNull { it.rawValue == raw } }
    }
}
