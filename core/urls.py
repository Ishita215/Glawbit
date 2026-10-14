from django.urls import path
from . import views

urlpatterns = [
    # Pages
    path("",                views.home,            name="home"),
    path("xray/",           views.xray_page,       name="xray"),
    path("market/",         views.market_page,     name="market"),
    path("evidence/",       views.evidence_page,   name="evidence"),

    # API
    path("analyze/",        views.analyze,             name="analyze"),
    path("api/market/",     views.api_market_overview, name="api_market"),
    path("api/health/",     views.api_data_health,     name="api_health"),
    path("api/evidence/",   views.api_model_evidence,  name="api_evidence"),
    path("api/command/",    views.api_command_centre,  name="api_command"),
]
