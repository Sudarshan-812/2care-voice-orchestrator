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

    async def get_practitioners(self) -> list[dict[str, Any]]:
        """Fetch active practitioners (doctors)."""
        data = await self._request("GET", "/practitioners")
        practitioners = data.get("practitioners", [])
        return [
            {
                "id": practitioner.get("id"),
                "first_name": practitioner.get("first_name"),
                "last_name": practitioner.get("last_name"),
            }
            for practitioner in practitioners
            if not practitioner.get("archived_at")
        ]

    async def get_available_times(
        self,
        business_id: int,
        practitioner_id: int,
        appointment_type_id: int,
        from_date: str,
        to_date: str,
    ) -> list[dict[str, Any]]:
        """Fetch real bookable slots from Cliniko's available_times endpoint.

        This reflects the practitioner's actual working hours and existing bookings —
        no gap-inference or mocking. from_date/to_date are plain dates (YYYY-MM-DD).
        """
        path = (
            f"/businesses/{business_id}/practitioners/{practitioner_id}"
            f"/appointment_types/{appointment_type_id}/available_times"
        )
        try:
            data = await self._request("GET", path, params={"from": from_date, "to": to_date})
        except ClinikoAPIError as exc:
            if exc.status_code == 404:
                # Cliniko 404s this endpoint (instead of returning an empty list) when the
                # practitioner has no working-hours schedule configured for this business/
                # appointment type combination. Treat that as "no slots" rather than
                # propagating a raw error the LLM tends to choke on mid tool-call generation.
                logger.info(
                    "No schedule found for %s (business=%s, practitioner=%s, appointment_type=%s); "
                    "treating as no available slots",
                    path,
                    business_id,
                    practitioner_id,
                    appointment_type_id,
                )
                return []
            raise
        return data.get("available_times", [])

    async def book_appointment(
        self,
        patient_id: int,
        practitioner_id: int,
        appointment_type_id: int,
        business_id: int,
        start_time: str,
        end_time: str,
    ) -> dict[str, Any]:
        """Book a new individual appointment."""
        payload = {
            "patient_id": patient_id,
            "practitioner_id": practitioner_id,
            "appointment_type_id": appointment_type_id,
            "business_id": business_id,
            "starts_at": start_time,
            "ends_at": end_time,
        }
        return await self._request("POST", "/individual_appointments", json=payload)

    async def cancel_appointment(self, appointment_id: int) -> bool:
        """Cancel (delete) an individual appointment. Returns True on success."""
        client = self.get_client()
        started_at = datetime.now(timezone.utc)
        start = time.monotonic()
        path = f"/individual_appointments/{appointment_id}"

        try:
            response = await client.request("DELETE", path)
        except httpx.RequestError as exc:
            elapsed_ms = (time.monotonic() - start) * 1000
            logger.error(
                "Cliniko API request failed: DELETE %s (%.1fms) at %s | error=%s",
                path,
                elapsed_ms,
                started_at.isoformat(),
                exc,
            )
            raise ClinikoAPIError(f"Cliniko API request failed for DELETE {path}: {exc}") from exc

        elapsed_ms = (time.monotonic() - start) * 1000
        logger.info(
            "Cliniko API DELETE %s -> %s (%.1fms) at %s",
            path,
            response.status_code,
            elapsed_ms,
            started_at.isoformat(),
        )

        if response.status_code in (200, 204):
            return True

        logger.error(
            "Cliniko API error: DELETE %s -> %s | body=%s", path, response.status_code, response.text
        )
        raise ClinikoAPIError(
            f"Cliniko API returned {response.status_code} for DELETE {path}",
            status_code=response.status_code,
        )

    async def get_patient_appointments(self, patient_id: int) -> dict[str, Any]:
        """Fetch a patient's individual appointments."""
        params = [("q[]", f"patient_id:{patient_id}")]
        return await self._request("GET", "/individual_appointments", params=params)

    async def get_patients_by_phone(self, phone_number: str) -> list[dict[str, Any]]:
        """Look up patients whose phone number contains the given number."""
        params = [("q[]", f"phone_numbers.number:contains:{phone_number}")]
        data = await self._request("GET", "/patients", params=params)
        patients = data.get("patients", [])
        return [
            {"id": patient.get("id"), "first_name": patient.get("first_name"), "last_name": patient.get("last_name")}
            for patient in patients
        ]

    async def get_patients_by_name(self, first_name: str, last_name: str) -> list[dict[str, Any]]:
        """Look up patients by first and last name (phone_numbers.number is not a filterable field)."""
        params = [
            ("q[]", f"first_name:{first_name}"),
            ("q[]", f"last_name:{last_name}"),
        ]
        data = await self._request("GET", "/patients", params=params)
        patients = data.get("patients", [])
        return [
            {"id": patient.get("id"), "first_name": patient.get("first_name"), "last_name": patient.get("last_name")}
            for patient in patients
        ]

    async def create_patient(self, first_name: str, last_name: str, phone_number: str) -> int:
        """Create a new patient record and return its Cliniko patient ID."""
        payload = {
            "first_name": first_name,
            "last_name": last_name,
            "patient_phone_numbers": [{"number": phone_number, "phone_type": "Mobile"}],
        }
        data = await self._request("POST", "/patients", json=payload)
        return data["id"]

    async def get_appointment_types(self) -> list[dict[str, Any]]:
        """Fetch active appointment types (clinic services)."""
        data = await self._request("GET", "/appointment_types")
        appointment_types = data.get("appointment_types", [])
        return [
            {
                "id": appointment_type.get("id"),
                "name": appointment_type.get("name"),
                "duration_in_minutes": appointment_type.get("duration_in_minutes"),
            }
            for appointment_type in appointment_types
            if not appointment_type.get("archived_at")
        ]

    async def get_businesses(self) -> list[dict[str, Any]]:
        """Fetch clinic locations (Cliniko 'Businesses')."""
        data = await self._request("GET", "/businesses")
        businesses = data.get("businesses", [])
        return [
            {
                "id": business.get("id"),
                "name": business.get("business_name"),
                "city": business.get("city"),
                "address": business.get("address_1"),
            }
            for business in businesses
        ]


cliniko_client = ClinikoClient()
