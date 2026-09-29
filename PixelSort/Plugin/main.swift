// Entry point of the embedded FxPlug service, not the visible wrapper application.
// FxPrincipal reads Info.plist registrations and accepts host requests for the three filter classes.
// Apple manages the service lifetime; starting the principal connects our callbacks to the host.
FxPrincipal.startServicePrincipal()
