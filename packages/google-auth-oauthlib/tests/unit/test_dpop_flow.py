# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import json
import os
from unittest import mock

import pytest
from cryptography.hazmat.primitives.asymmetric import ec

dpop = pytest.importorskip(
    "google.auth.dpop",
    reason="google-auth with DPoP support is required for these tests",
)
dpop_credentials = pytest.importorskip(
    "google.oauth2.dpop_credentials",
    reason="google-auth with DPoP support is required for these tests",
)
from google.oauth2 import credentials as _oauth2_credentials

from google_auth_oauthlib import flow, helpers

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CLIENT_SECRETS_FILE = os.path.join(DATA_DIR, "client_secrets.json")

with open(CLIENT_SECRETS_FILE, "r") as fh:
    CLIENT_SECRETS_INFO = json.load(fh)


@pytest.fixture
def dpop_signer():
    """Creates a real SoftwareDPoPSigner with a freshly generated EC key."""
    private_key = ec.generate_private_key(ec.SECP256R1())
    return dpop.SoftwareDPoPSigner(private_key)


class TestFlowWithDPoP(object):
    def test_from_client_config_with_dpop_signer(self, dpop_signer):
        instance = flow.Flow.from_client_config(
            CLIENT_SECRETS_INFO,
            scopes=mock.sentinel.scopes,
            dpop_signer=dpop_signer,
        )
        assert instance.dpop_signer is dpop_signer
        assert instance.client_config == CLIENT_SECRETS_INFO["web"]

    def test_from_client_config_without_dpop_signer(self):
        instance = flow.Flow.from_client_config(
            CLIENT_SECRETS_INFO,
            scopes=mock.sentinel.scopes,
        )
        assert instance.dpop_signer is None

    def test_from_client_secrets_file_with_dpop_signer(self, dpop_signer):
        instance = flow.Flow.from_client_secrets_file(
            CLIENT_SECRETS_FILE,
            scopes=mock.sentinel.scopes,
            dpop_signer=dpop_signer,
        )
        assert instance.dpop_signer is dpop_signer

    @pytest.fixture
    def instance(self, dpop_signer):
        yield flow.Flow.from_client_config(
            CLIENT_SECRETS_INFO,
            scopes=mock.sentinel.scopes,
            dpop_signer=dpop_signer,
        )

    def test_authorization_url_includes_dpop_jkt(self, instance, dpop_signer):
        instance.oauth2session.scope = ["scope_one"]
        authorization_url_patch = mock.patch.object(
            instance.oauth2session,
            "authorization_url",
            wraps=instance.oauth2session.authorization_url,
        )

        with authorization_url_patch as authorization_url_spy:
            url, _ = instance.authorization_url()

            assert "dpop_jkt=" in url
            _, kwargs = authorization_url_spy.call_args_list[0]
            assert kwargs["dpop_jkt"] == dpop_signer.thumbprint

    def test_authorization_url_no_dpop_jkt_without_signer(self):
        instance = flow.Flow.from_client_config(
            CLIENT_SECRETS_INFO,
            scopes=["scope_one"],
        )
        authorization_url_patch = mock.patch.object(
            instance.oauth2session,
            "authorization_url",
            wraps=instance.oauth2session.authorization_url,
        )

        with authorization_url_patch as authorization_url_spy:
            url, _ = instance.authorization_url()

            assert "dpop_jkt=" not in url
            _, kwargs = authorization_url_spy.call_args_list[0]
            assert "dpop_jkt" not in kwargs

    def test_fetch_token_injects_dpop_proof_header(self, instance, dpop_signer):
        instance.code_verifier = "amanaplanacanalpanama"

        fetch_token_patch = mock.patch.object(
            instance.oauth2session,
            "fetch_token",
            autospec=True,
            return_value=mock.sentinel.token,
        )

        with fetch_token_patch as fetch_token_mock:
            instance.fetch_token(code="test-auth-code")

            fetch_token_mock.assert_called_once()
            _, kwargs = fetch_token_mock.call_args
            headers = kwargs.get("headers", {})
            assert "DPoP" in headers

    def test_fetch_token_uses_auth_code_jti(self, instance, dpop_signer):
        """Verify the DPoP proof's jti is derived from the auth code."""
        instance.code_verifier = "verifier"

        captured_proofs = []
        original_create_proof = dpop_signer.create_proof

        def capturing_create_proof(*args, **kwargs):
            captured_proofs.append(kwargs)
            return original_create_proof(*args, **kwargs)

        with mock.patch.object(
            dpop_signer, "create_proof", side_effect=capturing_create_proof
        ):
            fetch_token_patch = mock.patch.object(
                instance.oauth2session,
                "fetch_token",
                autospec=True,
                return_value=mock.sentinel.token,
            )
            with fetch_token_patch:
                instance.fetch_token(code="my-auth-code")

        assert len(captured_proofs) == 1
        expected_jti = dpop.compute_auth_code_jti("my-auth-code")
        assert captured_proofs[0]["jti"] == expected_jti

    def test_fetch_token_extracts_code_from_authorization_response(
        self, instance, dpop_signer
    ):
        """Verify code is extracted from authorization_response URL."""
        instance.code_verifier = "verifier"

        captured_proofs = []
        original_create_proof = dpop_signer.create_proof

        def capturing_create_proof(*args, **kwargs):
            captured_proofs.append(kwargs)
            return original_create_proof(*args, **kwargs)

        with mock.patch.object(
            dpop_signer, "create_proof", side_effect=capturing_create_proof
        ):
            fetch_token_patch = mock.patch.object(
                instance.oauth2session,
                "fetch_token",
                autospec=True,
                return_value=mock.sentinel.token,
            )
            with fetch_token_patch:
                instance.fetch_token(
                    authorization_response=(
                        "https://localhost:8080/?code=extracted-code&state=xyz"
                    )
                )

        assert len(captured_proofs) == 1
        expected_jti = dpop.compute_auth_code_jti("extracted-code")
        assert captured_proofs[0]["jti"] == expected_jti

    def test_fetch_token_without_dpop_signer_no_dpop_header(self):
        """Without a signer, fetch_token should not add DPoP header."""
        instance = flow.Flow.from_client_config(
            CLIENT_SECRETS_INFO,
            scopes=mock.sentinel.scopes,
        )
        instance.code_verifier = "verifier"

        fetch_token_patch = mock.patch.object(
            instance.oauth2session,
            "fetch_token",
            autospec=True,
            return_value=mock.sentinel.token,
        )

        with fetch_token_patch as fetch_token_mock:
            instance.fetch_token(code="test-code")

            _, kwargs = fetch_token_mock.call_args
            # Without dpop_signer, no headers kwarg should be set
            # (or if set, no DPoP key in it)
            headers = kwargs.get("headers")
            if headers:
                assert "DPoP" not in headers

    def test_fetch_token_captures_dpop_nonce_from_response(self, instance, dpop_signer):
        """Verify the compliance hook captures DPoP-Nonce."""
        instance.code_verifier = "verifier"
        nonce_value = "server-nonce-abc"

        def fake_fetch_token(token_url, **kwargs):
            # Simulate what happens: compliance hooks are registered,
            # then fetch_token is called. We simulate the hook being
            # invoked by calling it manually.
            mock_response = mock.Mock()
            mock_response.headers = {"DPoP-Nonce": nonce_value}

            # The hook should have been registered on the session
            hooks = instance.oauth2session.compliance_hook["access_token_response"]
            for hook in hooks:
                hook(mock_response)

            return {
                "access_token": "test-token",
                "refresh_token": "test-refresh",
                "expires_at": 643969200.0,
            }

        with mock.patch.object(
            instance.oauth2session,
            "fetch_token",
            side_effect=fake_fetch_token,
        ):
            instance.fetch_token(code="test-code")

        assert instance.oauth2session._dpop_nonce == nonce_value

    def test_fetch_token_uses_nonce_from_session(self, instance, dpop_signer):
        """Verify that a nonce captured on the session is included in
        the next DPoP proof (not read from self/Flow)."""
        instance.code_verifier = "verifier"
        # Simulate a nonce previously captured by the compliance hook.
        instance.oauth2session._dpop_nonce = "previously-captured-nonce"

        captured_proofs = []
        original_create_proof = dpop_signer.create_proof

        def capturing_create_proof(*args, **kwargs):
            captured_proofs.append(kwargs)
            return original_create_proof(*args, **kwargs)

        with mock.patch.object(
            dpop_signer, "create_proof", side_effect=capturing_create_proof
        ):
            fetch_token_patch = mock.patch.object(
                instance.oauth2session,
                "fetch_token",
                autospec=True,
                return_value=mock.sentinel.token,
            )
            with fetch_token_patch:
                instance.fetch_token(code="test-code")

        assert len(captured_proofs) == 1
        assert captured_proofs[0]["nonce"] == "previously-captured-nonce"

    def test_credentials_returns_dpop_credentials(self, instance, dpop_signer):
        instance.oauth2session.token = {
            "access_token": "test-access-token",
            "refresh_token": "test-refresh-token",
            "id_token": "test-id-token",
            "expires_at": 643969200.0,
        }

        credentials = instance.credentials

        assert isinstance(credentials, dpop_credentials.DPoPCredentials)
        assert credentials.token == "test-access-token"
        assert credentials.expiry == datetime.datetime(1990, 5, 29, 8, 20, 0)
        assert credentials._refresh_token == "test-refresh-token"
        assert credentials.dpop_signer is dpop_signer

    def test_credentials_without_signer_returns_regular_credentials(self):
        instance = flow.Flow.from_client_config(
            CLIENT_SECRETS_INFO,
            scopes=mock.sentinel.scopes,
        )
        instance.oauth2session.token = {
            "access_token": "test-access-token",
            "refresh_token": "test-refresh-token",
            "id_token": "test-id-token",
            "expires_at": 643969200.0,
        }

        credentials = instance.credentials

        assert not isinstance(credentials, dpop_credentials.DPoPCredentials)
        assert isinstance(credentials, _oauth2_credentials.Credentials)

    def test_credentials_captures_nonce_from_session(self, instance, dpop_signer):
        """Verify that credentials pick up the DPoP-Nonce stored on the session."""
        instance.oauth2session.token = {
            "access_token": "test-access-token",
            "refresh_token": "test-refresh-token",
            "expires_at": 643969200.0,
        }
        instance.oauth2session._dpop_nonce = "captured-nonce"

        credentials = instance.credentials

        assert isinstance(credentials, dpop_credentials.DPoPCredentials)
        assert credentials.dpop_nonce == "captured-nonce"

    def test_compliance_hook_registered_only_once(self, instance, dpop_signer):
        """Multiple fetch_token calls must not accumulate duplicate hooks."""
        hooks_before = list(
            instance.oauth2session.compliance_hook.get("access_token_response", [])
        )
        num_before = len(hooks_before)

        fetch_token_patch = mock.patch.object(
            instance.oauth2session,
            "fetch_token",
            autospec=True,
            return_value={"access_token": "tok", "expires_at": 643969200.0},
        )

        with fetch_token_patch:
            instance.fetch_token(code="code1")
            instance.fetch_token(code="code2")
            instance.fetch_token(code="code3")

        hooks_after = list(
            instance.oauth2session.compliance_hook.get("access_token_response", [])
        )
        # Exactly one new hook should have been registered, not three.
        assert len(hooks_after) == num_before + 1


class TestInstalledAppFlowWithDPoP(object):
    SCOPES = ["email", "profile"]

    def test_from_client_secrets_file_with_dpop(self, dpop_signer):
        instance = flow.InstalledAppFlow.from_client_secrets_file(
            CLIENT_SECRETS_FILE,
            scopes=self.SCOPES,
            dpop_signer=dpop_signer,
        )
        assert isinstance(instance, flow.InstalledAppFlow)
        assert instance.dpop_signer is dpop_signer


class TestHelpersCredentialsFromSessionWithDPoP(object):
    @pytest.fixture
    def session(self):
        session, _ = helpers.session_from_client_config(
            CLIENT_SECRETS_INFO, scopes=mock.sentinel.scopes
        )
        yield session

    def test_credentials_from_session_with_dpop_signer(self, session, dpop_signer):
        session.token = {
            "access_token": "dpop-access-token",
            "refresh_token": "dpop-refresh-token",
            "id_token": "dpop-id-token",
            "expires_at": 643969200.0,
        }

        credentials = helpers.credentials_from_session(
            session, CLIENT_SECRETS_INFO["web"], dpop_signer=dpop_signer
        )

        assert isinstance(credentials, dpop_credentials.DPoPCredentials)
        assert credentials.token == "dpop-access-token"
        assert credentials.expiry == datetime.datetime(1990, 5, 29, 8, 20, 0)
        assert credentials._refresh_token == "dpop-refresh-token"
        assert credentials.id_token == "dpop-id-token"
        assert credentials._client_id == CLIENT_SECRETS_INFO["web"]["client_id"]
        assert credentials._client_secret == CLIENT_SECRETS_INFO["web"]["client_secret"]
        assert credentials._token_uri == CLIENT_SECRETS_INFO["web"]["token_uri"]
        assert credentials.dpop_signer is dpop_signer

    def test_credentials_from_session_without_dpop_signer(self, session):
        session.token = {
            "access_token": "bearer-token",
            "refresh_token": "bearer-refresh",
            "expires_at": 643969200.0,
        }

        credentials = helpers.credentials_from_session(
            session, CLIENT_SECRETS_INFO["web"]
        )

        assert not isinstance(credentials, dpop_credentials.DPoPCredentials)
        assert isinstance(
            credentials,
            _oauth2_credentials.Credentials,
        )

    def test_credentials_from_session_dpop_nonce_from_session(
        self, session, dpop_signer
    ):
        session.token = {
            "access_token": "dpop-token",
            "refresh_token": "dpop-refresh",
            "expires_at": 643969200.0,
        }
        session._dpop_nonce = "session-nonce"

        credentials = helpers.credentials_from_session(
            session, CLIENT_SECRETS_INFO["web"], dpop_signer=dpop_signer
        )

        assert isinstance(credentials, dpop_credentials.DPoPCredentials)
        assert credentials.dpop_nonce == "session-nonce"

    def test_credentials_from_session_dpop_with_granted_scopes(
        self, session, dpop_signer
    ):
        granted = ["scope1", "scope2"]
        session.token = {
            "access_token": "dpop-token",
            "refresh_token": "dpop-refresh",
            "expires_at": 643969200.0,
            "scope": granted,
        }

        credentials = helpers.credentials_from_session(
            session, CLIENT_SECRETS_INFO["web"], dpop_signer=dpop_signer
        )

        assert isinstance(credentials, dpop_credentials.DPoPCredentials)
        assert credentials.granted_scopes == granted
