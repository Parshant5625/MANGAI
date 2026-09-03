export function compactNumber(value: number, fractionDigits = 1): string {
  return new Intl.NumberFormat("en-IN", {
    notation: "compact",
    maximumFractionDigits: fractionDigits
  }).format(value);
}

export function number(value: number, fractionDigits = 0): string {
  return new Intl.NumberFormat("en-IN", {
    maximumFractionDigits: fractionDigits
  }).format(value);
}

export function percent(value: number, fractionDigits = 0): string {
  return `${(value * 100).toFixed(fractionDigits)}%`;
}

export function signedNumber(value: number): string {
  return `${value >= 0 ? "+" : ""}${number(value, 0)}`;
}

