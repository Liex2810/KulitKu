import 'package:flutter_test/flutter_test.dart';
import 'package:kulitku/main.dart';

void main() {
  testWidgets('local demo clearly labels scope and scan controls', (
    tester,
  ) async {
    await tester.pumpWidget(const KulitKuApp());
    await tester.pumpAndSettle();
    expect(find.text('KulitKu Lab'), findsOneWidget);
    expect(find.text('Mode lokal / uji coba'), findsOneWidget);
    expect(find.text('Analisis foto'), findsOneWidget);
    expect(find.textContaining('Prototipe penelitian'), findsOneWidget);
  });
}
