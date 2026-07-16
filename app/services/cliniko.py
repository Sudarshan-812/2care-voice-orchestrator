import logging
import time
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

USER_AGENT = "2care-Voice-Orchestrator (tech@2care.ai)"


class ClinikoAPIError(Exception):
    """Raised when the Cliniko API returns an error or is unreachable."""

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class ClinikoClient:
    """Async wrapper around the Cliniko REST API."""

    def __init__(self) -> None:
        self.api_key = settings.CLINIKO_API_KEY
        self.shard = settings.CLINIKO_SHARD
        self.base_url = f"https://api.{self.shard}.cliniko.com/v1"
        self._client: Optional[httpx.AsyncClient] = None

    def _build_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.base_url,
            auth=httpx.BasicAuth(self.api_key, ""),
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "application/json",
            },
            timeout=30.0,
        )

    def get_client(self) -> httpx.AsyncClient:
        """Lazily create and return the underlying httpx.AsyncClient."""
        if self._client is None:
            self._client = self._build_client()
        return self._client

    async def __aenter__(self) -> "ClinikoClient":
        self._client = self._build_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        client = self.get_client()
        started_at = datetime.now(timezone.utc)
        start = time.monotonic()

        try:
            response = await client.request(method, path, **kwargs)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            elapsed_ms = (time.monotonic() - start) * 1000
            logger.error(
                "Cliniko API error: %s %s -> %s (%.1fms) at %s | body=%s",
                method,
                path,
                exc.response.status_code,
                elapsed_ms,
                started_at.isoformat(),
                exc.response.text,
            )
            raise ClinikoAPIError(
                f"Cliniko API returned {exc.response.status_code} for {method} {path}",
                status_code=exc.response.status_code,
            ) from exc
        except httpx.RequestError as exc:
            elapsed_ms = (time.monotonic() - start) * 1000
            logger.error(
                "Cliniko API request failed: %s %s (%.1fms) at %s | error=%s",
                method,
                path,
                elapsed_ms,
                started_at.isoformat(),
                exc,
            )
            raise ClinikoAPIError(f"Cliniko API request failed for {method} {path}: {exc}") from exc

        elapsed_ms = (time.monotonic() - start) * 1000
        logger.info(
            "Cliniko API %s %s -> %s (%.1fms) at %s",
            method,
            path,
            response.status_code,
            elapsed_ms,
            started_at.isoformat(),
        )
        return response.json()

    async def get_practitioners(self) -> dict[str, Any]:
        """Fetch the list of practitioners (doctors)."""
        return await self._request("GET", "/practitioners")

    async def get_appointments(self, start_date: str, end_date: str) -> dict[str, Any]:
        """Fetch individual appointments starting after start_date and ending before end_date."""
        params = [
            ("q[]", f"starts_at:>{start_date}"),
            ("q[]", f"ends_at:<{end_date}"),
        ]
        return await self._request("GET", "/individual_appointments", params=params)

    async def book_appointment(
        self,
        patient_id: int,
        practitioner_id: int,
        appointment_type_id: int,
        start_time: str,
        end_time: str,
    ) -> dict[str, Any]:
        """Book a new individual appointment."""
        payload = {
            "patient_id": patient_id,
            "practitioner_id": practitioner_id,
            "appointment_type_id": appointment_type_id,
            "starts_at": start_time,
            "ends_at": end_time,
        }
        return await self._request("POST", "/individual_appointments", json=payload)


cliniko_client = ClinikoClient()
